# =========================
# src/model_utils_bank_opt_multi.py
# (No cache, switchable single/multi real-image DA, selectable feature layer)
# =========================
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torchvision.transforms as T
from torchvision.utils import save_image
from transformers import CLIPVisionModel, Dinov2Model, ViTModel, SiglipVisionModel

try:
    import timm
except ImportError:
    timm = None


# ==========================================================
# 1. 基本ユーティリティ & Loss
# ==========================================================
def manage_model_allocation(models, weights, device):
    for i, model in enumerate(models):
        if weights[i] > 0:
            model.to(device)
        else:
            model.cpu()


class RandomGaussianNoise(nn.Module):
    def __init__(self, mean=0.0, std=1.0, p=0.5):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, images):
        if self.p == 0 or torch.rand(1).item() >= self.p:
            return images
        noise = torch.randn_like(images) * self.std + self.mean
        return images + noise


class TVLoss(nn.Module):
    def forward(self, img):
        _b, _c, h, w = img.size()
        h_tv = torch.pow((img[:, :, 1:, :] - img[:, :, : h - 1, :]), 2).mean()
        w_tv = torch.pow((img[:, :, :, 1:] - img[:, :, :, : w - 1]), 2).mean()
        return h_tv + w_tv


# ==========================================================
# 2. モデル定義 (EncoderClassifier)
#    - feature_layer を追加
#    - hidden_states から中間層特徴を抽出可能に変更
# ==========================================================
class EncoderClassifier(nn.Module):
    def __init__(
        self,
        encoder_model="openai/clip-vit-base-patch16",
        encoder=None,
        freeze_encoder=True,
        num_classes=1000,
        feature_source="pooler",
        projection_dim=2048,
        feature_layer=-1,
    ):
        super().__init__()
        self.encoder_model_name = encoder_model
        self.use_timm_encoder = False
        self.feature_layer = feature_layer

        if encoder is not None:
            self.encoder = encoder
            if not hasattr(encoder, "config"):
                self.use_timm_encoder = True
        else:
            if "siglip" in self.encoder_model_name:
                self.encoder = SiglipVisionModel.from_pretrained(encoder_model)
            elif self.encoder_model_name.startswith("openai/clip") or "laion" in self.encoder_model_name:
                self.encoder = CLIPVisionModel.from_pretrained(encoder_model)
            elif "dinov2" in self.encoder_model_name and "timm" not in self.encoder_model_name:
                self.encoder = Dinov2Model.from_pretrained(encoder_model)
            elif (
                "dino" in self.encoder_model_name
                and "v3" not in self.encoder_model_name
                and "v2" not in self.encoder_model_name
                and "timm" not in self.encoder_model_name
            ):
                self.encoder = ViTModel.from_pretrained(encoder_model)
            elif "dinov3" in self.encoder_model_name or self.encoder_model_name.startswith("timm/"):
                if timm is None:
                    raise ImportError("timm library is required for DINOv3 or timm models.")
                model_name = encoder_model.replace("timm/", "")
                self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
                self.use_timm_encoder = True
            else:
                try:
                    self.encoder = ViTModel.from_pretrained(encoder_model)
                except Exception:
                    raise ValueError(f"Unsupported encoder model: {encoder_model}")

        if self.use_timm_encoder:
            self.embed_dim = self.encoder.num_features
            self.feature_source = feature_source
        else:
            if "siglip" in self.encoder_model_name:
                self.feature_source = "mean" if feature_source == "pooler" else feature_source
            elif self.encoder_model_name.startswith("openai/clip"):
                self.feature_source = feature_source if feature_source in ["pooler", "cls", "mean"] else "pooler"
            elif "dinov2" in self.encoder_model_name:
                self.feature_source = "cls" if feature_source == "pooler" else feature_source
            elif "dino" in self.encoder_model_name:
                if feature_source == "pooler" and not hasattr(self.encoder.config, "pooler_type"):
                    self.feature_source = "cls"
                else:
                    self.feature_source = feature_source
            else:
                self.feature_source = feature_source

            cfg = self.encoder.config
            self.embed_dim = cfg.hidden_size

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if projection_dim > 0:
            self.projector = nn.Sequential(
                nn.Linear(self.embed_dim, projection_dim),
                nn.ReLU(inplace=True)
            )
            classifier_in_dim = projection_dim
        else:
            self.projector = nn.Identity()
            classifier_in_dim = self.embed_dim

        self.classifier = nn.Linear(classifier_in_dim, num_classes)
        self.reset_classifier()

    def reset_classifier(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

    def _extract_features_from_hidden(self, hidden_state):
        # hidden_state: [B, T, D]
        # 中間層では pooler が通常ないため, pooler 指定時は cls 扱いに寄せる
        if self.feature_source == "mean":
            if hidden_state.size(1) > 1:
                return hidden_state[:, 1:, :].mean(dim=1)
            return hidden_state[:, 0, :]
        return hidden_state[:, 0, :]

    def _extract_features_internal(self, outputs):
        if isinstance(outputs, torch.Tensor):
            return outputs

        if "siglip" in self.encoder_model_name:
            if self.feature_source == "pooler" and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            if self.feature_source == "mean":
                return outputs.last_hidden_state.mean(dim=1)
            return outputs.last_hidden_state[:, 0, :]

        if self.encoder_model_name.startswith("openai/clip") or "laion" in self.encoder_model_name:
            if self.feature_source == "pooler" and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            if self.feature_source == "mean":
                if outputs.last_hidden_state.size(1) > 1:
                    return outputs.last_hidden_state[:, 1:, :].mean(dim=1)
                return outputs.last_hidden_state[:, 0, :]
            return outputs.last_hidden_state[:, 0, :]

        if "dino" in self.encoder_model_name:
            if self.feature_source == "mean":
                if outputs.last_hidden_state.size(1) > 1:
                    return outputs.last_hidden_state[:, 1:, :].mean(dim=1)
                return outputs.last_hidden_state[:, 0, :]
            return outputs.last_hidden_state[:, 0, :]

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None and self.feature_source == "pooler":
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0, :]

    def _forward_features_timm(self, x):
        # timm では全層 hidden_states の統一取得が難しいため, 現状は最終特徴のみ対応
        outputs = self.encoder(x)
        if isinstance(outputs, torch.Tensor):
            if outputs.ndim == 3:
                return self._extract_features_from_hidden(outputs)
            return outputs
        return outputs

    def forward_features(self, x):
        if self.use_timm_encoder:
            return self._forward_features_timm(x)

        need_hidden = (self.feature_layer != -1)
        outputs = self.encoder(
            pixel_values=x,
            output_hidden_states=need_hidden
        )

        if self.feature_layer == -1:
            return self._extract_features_internal(outputs)

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("hidden_states is None. This model may not support output_hidden_states.")
        if not (-len(hidden_states) <= self.feature_layer < len(hidden_states)):
            raise ValueError(
                f"feature_layer={self.feature_layer} is out of range. "
                f"Available range: [{-len(hidden_states)}, {len(hidden_states)-1}]"
            )

        selected_hidden = hidden_states[self.feature_layer]
        return self._extract_features_from_hidden(selected_hidden)

    def forward_head(self, features):
        projected_features = self.projector(features)
        logits = self.classifier(projected_features)
        return logits

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


# ==========================================================
# 3. 生成器 (PyramidGenerator)
# ==========================================================
class PyramidGenerator(nn.Module):
    def __init__(
        self,
        target_size=224,
        start_size=16,
        activation="sigmoid",
        initial_image=None,
        noise_level=0.0,
        batch_size=1,
    ):
        super().__init__()
        self.target_size = target_size
        self.activation = activation
        self.batch_size = batch_size

        color_correlation_svd_sqrt = torch.tensor(
            [
                [0.26, 0.09, 0.02],
                [0.27, 0.00, -0.05],
                [0.27, -0.09, 0.03],
            ]
        )
        self.register_buffer("color_correlation", color_correlation_svd_sqrt)
        max_norm = torch.max(torch.linalg.norm(color_correlation_svd_sqrt, dim=0))
        self.register_buffer("max_norm", max_norm)

        normalized_matrix = color_correlation_svd_sqrt / max_norm
        try:
            inverse_matrix = torch.linalg.inv(normalized_matrix)
        except RuntimeError:
            inverse_matrix = torch.linalg.pinv(normalized_matrix)
        self.register_buffer("inverse_color_correlation", inverse_matrix)

        if initial_image is not None:
            init_low_res = torch.nn.functional.interpolate(
                initial_image,
                size=(start_size, start_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            eps = 1e-6
            if activation == "sigmoid":
                init_low_res = init_low_res.clamp(eps, 1 - eps)
                init_val = torch.logit(init_low_res) / 2
            else:
                init_val = init_low_res
            init_val = self.inverse_linear_decorrelate_color(init_val)
            if noise_level > 0:
                noise = torch.randn_like(init_val) * 0.1
                init_val = init_val * (1 - noise_level) + noise * noise_level

            init_val = init_val.repeat(self.batch_size, 1, 1, 1)
            self.levels = nn.ParameterList([nn.Parameter(init_val)])
        else:
            self.levels = nn.ParameterList(
                [nn.Parameter(torch.randn(self.batch_size, 3, start_size, start_size) * 0.1)]
            )

    def extend(self):
        current_res = max([p.shape[-1] for p in self.levels])
        if current_res >= self.target_size:
            return False
        new_res = min(current_res * 2, self.target_size)
        old_len = len(self.levels)
        new_len = old_len + 1
        with torch.no_grad():
            for p in self.levels:
                p.mul_(old_len / new_len)
        device = next(self.parameters()).device
        new_level = nn.Parameter(
            torch.randn(self.batch_size, 3, new_res, new_res).to(device) * (1.0 / new_len)
        )
        self.levels.append(new_level)
        return True

    def linear_decorrelate_color(self, t):
        t_permute = t.permute(0, 2, 3, 1)
        t_matched = torch.matmul(t_permute, self.color_correlation.T)
        t_matched = t_matched / self.max_norm
        return t_matched.permute(0, 3, 1, 2)

    def inverse_linear_decorrelate_color(self, t):
        t_permute = t.permute(0, 2, 3, 1)
        t_inverted = torch.matmul(t_permute, self.inverse_color_correlation.T)
        return t_inverted.permute(0, 3, 1, 2)

    def forward(self):
        device = next(self.parameters()).device
        image = torch.zeros(self.batch_size, 3, self.target_size, self.target_size).to(device)

        for level_tensor in self.levels:
            upsampled = torch.nn.functional.interpolate(
                level_tensor,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            image = image + upsampled

        image = self.linear_decorrelate_color(image)
        if self.activation == "sigmoid":
            return torch.sigmoid(2 * image)
        return image


# ==========================================================
# 4. FeatureBank 管理機能
#    - 構造維持のためクラス名は残す
#    - ただし cache は作らない
# ==========================================================
class FeatureBankSystem:
    def __init__(self, args, device, cache_dir="./feature_cache"):
        self.args = args
        self.device = device
        self.cache_dir = cache_dir
        self.register_normalization_buffers()

    def register_normalization_buffers(self):
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def preprocess(self, img):
        return (img - self.norm_mean) / self.norm_std


# ==========================================================
# 5. メイン最適化クラス
#    - cache なし
#    - single / multi real-image DA 切替
#    - 選択層特徴に対応
# ==========================================================
class MultiModelGM:
    def __init__(self, models, model_weights, target_classes, args, device, initial_image=None):
        self.models = models
        self.model_weights = model_weights
        self.target_classes = target_classes
        self.batch_size = len(target_classes)
        self.args = args
        self.device = device

        self.generator = PyramidGenerator(
            target_size=args.image_size,
            start_size=args.pyramid_start_res,
            activation="sigmoid",
            initial_image=initial_image,
            noise_level=args.seed_noise_level,
            batch_size=self.batch_size,
        ).to(device)

        self.optimizer = self._init_optimizer()

        self.syn_augmentor = T.Compose(
            [
                T.RandomResizedCrop(args.image_size, scale=(args.min_scale, args.max_scale), antialias=True),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                T.ColorJitter(0.1, 0.1, 0.1, 0.05),
                RandomGaussianNoise(mean=0.0, std=args.noise_std, p=args.noise_prob),
            ]
        )

        self.real_augmentor = T.Compose(
            [
                T.RandomResizedCrop(args.image_size, scale=(args.min_scale, args.max_scale), antialias=True),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(0.1, 0.1, 0.1, 0.05),
                RandomGaussianNoise(mean=0.0, std=args.noise_std, p=args.noise_prob),
            ]
        )

        if torch.cuda.is_available():
            try:
                self.scaler = torch.amp.GradScaler("cuda")
            except Exception:
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.tv_loss_fn = TVLoss().to(device)
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        self.syn_aug_num = int(self.args.syn_aug)

    def _init_optimizer(self):
        return optim.Adam(self.generator.parameters(), lr=self.args.lr)

    def preprocess(self, img):
        return (img - self.norm_mean) / self.norm_std

    def get_grads_from_features(self, model, features, tgt_class, create_graph=False):
        params = list(model.classifier.parameters())
        logits = model.forward_head(features)
        target_labels = torch.full((features.size(0),), int(tgt_class), device=self.device, dtype=torch.long)
        loss = F.cross_entropy(logits, target_labels)
        grads = torch.autograd.grad(loss, params, create_graph=create_graph)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])
        return flat_grad

    def _sample_real_images_from_pool(self, real_image_pool_cpu):
        # real_image_pool_cpu: [N, 3, H, W] on CPU
        if real_image_pool_cpu is None or real_image_pool_cpu.numel() == 0:
            raise ValueError("real_image_pool_cpu is empty.")

        n_pool = real_image_pool_cpu.size(0)

        if self.args.real_sampling_mode == "single":
            sample_count = 1
        else:
            sample_count = max(1, int(self.args.real_images_per_step))

        replace = (n_pool < sample_count)
        if replace:
            idx = torch.randint(0, n_pool, (sample_count,))
        else:
            idx = torch.randperm(n_pool)[:sample_count]

        selected = real_image_pool_cpu.index_select(0, idx)
        return selected

    def _build_real_batch(self, real_image_pool_cpu, target_count):
        """
        real_image_pool_cpu から single / multi を切り替えて画像を選び,
        DA をまとめて一括適用し,
        最終的に target_count 枚に揃えた real batch を返す.
        """
        selected = self._sample_real_images_from_pool(real_image_pool_cpu)  # [K, 3, H, W]
        aug_per_image = max(1, int(self.args.real_aug_per_image))

        # CPU -> GPU をまとめて1回
        selected = selected.to(self.device, non_blocking=True)  # [K, 3, H, W]

        # 各画像を aug_per_image 回だけ複製
        # [K, 3, H, W] -> [K*A, 3, H, W]
        real_batch = selected.repeat_interleave(aug_per_image, dim=0)

        # DA を一括で適用
        real_batch = self.real_augmentor(real_batch)

        # syn_aug と数を合わせる
        current_count = real_batch.size(0)
        if current_count < target_count:
            extra_idx = torch.randint(0, current_count, (target_count - current_count,), device=self.device)
            extra = real_batch.index_select(0, extra_idx)
            real_batch = torch.cat([real_batch, extra], dim=0)
        elif current_count > target_count:
            pick = torch.randperm(current_count, device=self.device)[:target_count]
            real_batch = real_batch.index_select(0, pick)

        return real_batch

        """
        real_image_pool_cpu から single / multi を切り替えて画像を選び,
        DA をかけ, 最終的に target_count 枚に揃えた real batch を返す.
        """
        selected = self._sample_real_images_from_pool(real_image_pool_cpu)  # [K, 3, H, W]
        aug_per_image = max(1, int(self.args.real_aug_per_image))

        aug_list = []
        for k in range(selected.size(0)):
            img = selected[k].unsqueeze(0).repeat(aug_per_image, 1, 1, 1).to(self.device, non_blocking=True)
            aug_img = self.real_augmentor(img)
            aug_list.append(aug_img)

        real_batch = torch.cat(aug_list, dim=0)  # [K*A, 3, H, W]

        # syn_aug と数を合わせる
        if real_batch.size(0) < target_count:
            extra_idx = torch.randint(0, real_batch.size(0), (target_count - real_batch.size(0),), device=self.device)
            extra = real_batch.index_select(0, extra_idx)
            real_batch = torch.cat([real_batch, extra], dim=0)
        elif real_batch.size(0) > target_count:
            pick = torch.randperm(real_batch.size(0), device=self.device)[:target_count]
            real_batch = real_batch.index_select(0, pick)

        return real_batch

    def _extract_real_features(self, model, real_image_pool_cpu, target_count):
        real_batch = self._build_real_batch(real_image_pool_cpu, target_count=target_count)
        inp_real = self.preprocess(real_batch)

        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                real_feats = model.forward_features(inp_real)
        except Exception:
            with autocast():
                real_feats = model.forward_features(inp_real)

        return real_feats

    def optimize_step(self, real_image_pools_cpu):
        self.optimizer.zero_grad()
        syn_image = self.generator()  # [B, 3, H, W]

        augmented_list = []
        for _ in range(self.syn_aug_num):
            augmented_list.append(self.syn_augmentor(syn_image))

        augmented_batch = torch.stack(augmented_list, dim=1).reshape(
            -1, 3, self.args.image_size, self.args.image_size
        )
        inp_syn = self.preprocess(augmented_batch)

        total_grad_loss = 0.0
        per_model_sims = {
            self.args.encoder_names[i]: 0.0
            for i in range(len(self.models))
            if self.model_weights[i] > 0
        }

        for i, model in enumerate(self.models):
            weight = self.model_weights[i]
            if weight == 0:
                continue

            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    syn_feats_all = model.forward_features(inp_syn)
            except Exception:
                with autocast():
                    syn_feats_all = model.forward_features(inp_syn)

            for b_idx, tgt_cls in enumerate(self.target_classes):
                model.reset_classifier()

                start_idx = b_idx * self.syn_aug_num
                end_idx = start_idx + self.syn_aug_num
                syn_feats = syn_feats_all[start_idx:end_idx]

                real_feats = self._extract_real_features(
                    model=model,
                    real_image_pool_cpu=real_image_pools_cpu[b_idx],
                    target_count=self.syn_aug_num,
                )

                try:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        target_grad = self.get_grads_from_features(
                            model, real_feats, tgt_cls, create_graph=False
                        )
                        syn_grad = self.get_grads_from_features(
                            model, syn_feats, tgt_cls, create_graph=True
                        )
                except Exception:
                    with autocast():
                        target_grad = self.get_grads_from_features(
                            model, real_feats, tgt_cls, create_graph=False
                        )
                        syn_grad = self.get_grads_from_features(
                            model, syn_feats, tgt_cls, create_graph=True
                        )

                sim = F.cosine_similarity(
                    target_grad.unsqueeze(0).detach(),
                    syn_grad.unsqueeze(0)
                ).mean()

                loss_k = 1.0 - sim
                total_grad_loss += (loss_k * weight) / self.batch_size

                model_name = self.args.encoder_names[i] if i < len(self.args.encoder_names) else f"model_{i}"
                per_model_sims[model_name] += sim.item() / self.batch_size

        loss_tv = self.tv_loss_fn(syn_image)
        total_loss = (total_grad_loss * self.args.weight_grad) + (loss_tv * self.args.weight_tv)

        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        return total_grad_loss.item(), loss_tv.item(), per_model_sims, total_loss.item()

    def run(self, real_image_pools_cpu, save_dirs, class_names, logger, global_pbar, gen_idx=0):
        logger.log(
            f"[Batch Size {self.batch_size}][Gen {gen_idx}] Optimization Start. "
            f"real_sampling_mode={self.args.real_sampling_mode}, "
            f"feature_layer={self.args.feature_layer}"
        )

        loss_history = []
        best_loss = float("inf")
        best_img_tensor = None

        from tqdm import tqdm
        local_pbar = tqdm(range(int(self.args.num_iterations)), desc=f"G{gen_idx}", leave=False)
        for i in local_pbar:
            if i > 0 and i % int(self.args.pyramid_grow_interval) == 0:
                if self.generator.extend():
                    self.optimizer = self._init_optimizer()

            l_grad, l_tv, model_sims, current_total_loss = self.optimize_step(real_image_pools_cpu)

            if current_total_loss < best_loss:
                best_loss = current_total_loss
                best_img_tensor = self.generator().detach().cpu()

            step_metrics = {
                "loss_grad": l_grad,
                "loss_tv": l_tv,
                "total_loss": current_total_loss,
            }
            for m_name, m_sim in model_sims.items():
                step_metrics[f"sim_{m_name}"] = m_sim
            loss_history.append(step_metrics)

            local_pbar.set_description(f"G{gen_idx} L:{l_grad:.3f}")
            global_pbar.update(self.batch_size)

            if i % 500 == 0:
                with torch.no_grad():
                    current_imgs = self.generator().detach().cpu()
                    for b_idx, s_dir in enumerate(save_dirs):
                        save_image(
                            current_imgs[b_idx].unsqueeze(0),
                            os.path.join(s_dir, f"step_{i:04d}_gen{gen_idx:02d}.png"),
                        )

        final_img = self.generator().detach().cpu()
        return final_img, best_img_tensor, {"loss_history": loss_history}