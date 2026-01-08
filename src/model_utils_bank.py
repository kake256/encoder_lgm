# =========================
# model_utils_bank.py
# (Priority A: Performance & Memory Optimized)
# =========================
import os
import re
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torchvision.transforms as T
from torchvision.utils import save_image
from transformers import CLIPVisionModel, Dinov2Model, ViTModel, SiglipVisionModel
from tqdm import tqdm

try:
    import timm
except ImportError:
    timm = None


# ==========================================================
# 1. 基本ユーティリティ & Loss
# ==========================================================
def manage_model_allocation(models, weights, device):
    """
    重みが0のモデルをCPUに退避し、必要なモデルのみGPUに配置する。
    頻繁な empty_cache() は呼び出し側で制御するため、ここでは最小限にする。
    """
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
        # メモリ効率のため、in-place加算を行わないように注意しつつ生成
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
    ):
        super().__init__()
        self.encoder_model_name = encoder_model
        self.use_timm_encoder = False

        # --- Encoder Initialization (変更なし) ---
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
            # Hugging Face models setup
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
            self.projector = nn.Sequential(nn.Linear(self.embed_dim, projection_dim), nn.ReLU(inplace=True))
            classifier_in_dim = projection_dim
        else:
            self.projector = nn.Identity()
            classifier_in_dim = self.embed_dim

        self.classifier = nn.Linear(classifier_in_dim, num_classes)
        self.reset_classifier()

    def reset_classifier(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

    def _extract_features_internal(self, outputs):
        if isinstance(outputs, torch.Tensor):
            return outputs

        if "siglip" in self.encoder_model_name:
            if self.feature_source == "pooler" and hasattr(outputs, "pooler_output"):
                return outputs.pooler_output
            if self.feature_source == "mean":
                return outputs.last_hidden_state.mean(dim=1)
            return outputs.last_hidden_state[:, 0, :]

        if self.encoder_model_name.startswith("openai/clip"):
            if self.feature_source == "pooler":
                return outputs.pooler_output
            if self.feature_source == "cls":
                return outputs.last_hidden_state[:, 0, :]
            if self.feature_source == "mean":
                return outputs.last_hidden_state[:, 1:, :].mean(dim=1)

        if "dino" in self.encoder_model_name:
            if self.feature_source == "cls":
                return outputs.last_hidden_state[:, 0, :]
            if self.feature_source == "mean":
                return outputs.last_hidden_state[:, 1:, :].mean(dim=1)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0, :]

    def forward_features(self, x):
        if self.use_timm_encoder:
            outputs = self.encoder(x)
        else:
            outputs = self.encoder(pixel_values=x, output_hidden_states=False)
        return self._extract_features_internal(outputs)

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
    def __init__(self, target_size=224, start_size=16, activation="sigmoid", initial_image=None, noise_level=0.0):
        super().__init__()
        self.target_size = target_size
        self.activation = activation

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
            # 初期画像がある場合
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
            self.levels = nn.ParameterList([nn.Parameter(init_val)])
        else:
            # ランダム初期化
            self.levels = nn.ParameterList([nn.Parameter(torch.randn(1, 3, start_size, start_size) * 0.1)])

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
        new_level = nn.Parameter(torch.randn(1, 3, new_res, new_res).to(device) * (1.0 / new_len))
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
        image = torch.zeros(1, 3, self.target_size, self.target_size).to(device)
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
# 4. Feature Bank 管理機能
# ==========================================================
class FeatureBankSystem:
    def __init__(self, args, device, cache_dir="./feature_cache"):
        self.args = args
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Augmentor for Real Images
        self.real_augmentor = T.Compose(
            [
                T.RandomResizedCrop(args.image_size, scale=(args.min_scale, args.max_scale), antialias=True),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(0.1, 0.1, 0.1, 0.05),
                RandomGaussianNoise(mean=0.0, std=args.noise_std, p=args.noise_prob),
            ]
        )

        # Pre-register normalization constants to avoid repeated tensor creation
        self.register_normalization_buffers()

    def register_normalization_buffers(self):
        # ImageNet normalization
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def preprocess(self, img):
        # 毎回 Tensor を作らず、登録済みの buffer を使う
        return (img - self.norm_mean) / self.norm_std

    @staticmethod
    def _safe_token(s: str) -> str:
        if s is None:
            return "none"
        s2 = str(s).strip()
        if s2 == "":
            return "none"
        s2 = re.sub(r"[^a-zA-Z0-9]", "_", s2)
        s2 = re.sub(r"_+", "_", s2).strip("_")
        return s2 if s2 else "none"

    @staticmethod
    def _fmt_float(x: float, ndigits: int = 3) -> str:
        try:
            return f"{float(x):.{ndigits}f}"
        except Exception:
            return "nan"

    def get_cache_path(self, target_class: int, model_name: str):
        dset = self._safe_token(self.args.dataset_type)
        split = self._safe_token(self.args.dataset_split)
        ov_txt = self._safe_token(self.args.overlay_text)
        ov_col = self._safe_token(self.args.text_color)
        ov_fnt = self._fmt_float(self.args.font_scale, 3)
        overlay_dir = f"overlay_{ov_txt}_{ov_col}_f{ov_fnt}"
        img_dir = f"img{int(self.args.image_size)}"
        sc_dir = f"sc{self._fmt_float(self.args.min_scale,3)}-{self._fmt_float(self.args.max_scale,3)}"
        ns_dir = f"ns{self._fmt_float(self.args.noise_prob,3)}-{self._fmt_float(self.args.noise_std,3)}"
        aug_dir = f"{sc_dir}_{ns_dir}"
        use_real_dir = f"use_real{int(self.args.use_real)}_raug{int(self.args.real_aug)}"
        safe_model = self._safe_token(model_name)

        subdir = os.path.join(
            self.cache_dir, dset, split, overlay_dir, img_dir, aug_dir, use_real_dir,
            f"cls_{int(target_class)}", f"model_{safe_model}"
        )
        os.makedirs(subdir, exist_ok=True)

        info_parts = [
            f"dset_{dset}", f"split_{split}", f"cls_{int(target_class)}",
            f"overlay_text_{ov_txt}", f"text_color_{ov_col}", f"font_scale_{ov_fnt}",
            f"image_size_{int(self.args.image_size)}", f"use_real_{int(self.args.use_real)}",
            f"real_aug_{int(self.args.real_aug)}", f"min_scale_{self._fmt_float(self.args.min_scale,6)}",
            f"max_scale_{self._fmt_float(self.args.max_scale,6)}",
            f"noise_prob_{self._fmt_float(self.args.noise_prob,6)}",
            f"noise_std_{self._fmt_float(self.args.noise_std,6)}",
            f"model_{model_name}",
        ]
        params_hash = hashlib.md5("_".join(info_parts).encode()).hexdigest()
        return os.path.join(subdir, f"features_{params_hash}.pt")

    def create_or_load_bank(self, models, raw_images_pool_cpu: torch.Tensor, target_class: int, logger):
        """
        全モデル分のFeature Bankをリストで返す.
        Priority A: Memory Pre-allocation & Efficient Batching
        """
        bank_list = []
        logger.log(f"Preparing Feature Bank for Class {target_class}...")

        if raw_images_pool_cpu.device.type != "cpu":
            raw_images_pool_cpu = raw_images_pool_cpu.cpu()

        if int(self.args.use_real) > 0:
            raw_images_pool_cpu = raw_images_pool_cpu[: int(self.args.use_real)].contiguous()

        num_images = len(raw_images_pool_cpu)
        real_aug = int(self.args.real_aug)
        
        # OOM回避: 固定32ではなく引数から取得、なければデフォルト
        batch_size = getattr(self.args, "batch_size", 32)
        total_vectors = num_images * real_aug

        for i, model in enumerate(models):
            model_name = self.args.encoder_names[i]
            cache_path = self.get_cache_path(target_class, model_name)

            if os.path.exists(cache_path):
                # 破損チェック：ロードして失敗したら再計算
                try:
                    feats_cpu = torch.load(cache_path, map_location="cpu")
                    logger.log(f"  [Cache Hit] {model_name} -> {cache_path}")
                    bank_list.append(feats_cpu)
                    continue
                except Exception as e:
                    logger.log(f"  [Cache Error] Failed to load {cache_path}: {e}. Recomputing...")

            logger.log(f"  [Cache Miss] Computing features for {model_name}...")
            
            # --- 埋め込み次元の特定 (ダミーフォワード) ---
            model_device = next(model.parameters()).device
            if model_device.type != "cuda" and torch.cuda.is_available():
                model.to(self.device)
            
            # メモリ確保: CPU上に巨大Tensorを事前に確保し、断片化を防ぐ
            # shape: (N * Aug, EmbedDim)
            embed_dim = model.embed_dim
            final_bank_tensor = torch.zeros((total_vectors, embed_dim), dtype=torch.float32, device="cpu")
            
            fill_pointer = 0
            
            # Augmentation loop
            with torch.no_grad():
                with tqdm(total=total_vectors, desc=f"  Comp {model_name}", leave=True, unit="img") as pbar:
                    for _ in range(real_aug):
                        # 画像バッチループ
                        for start in range(0, num_images, batch_size):
                            end = min(start + batch_size, num_images)
                            current_batch_size = end - start
                            
                            # 画像転送: CPU -> GPU
                            batch_cpu = raw_images_pool_cpu[start:end]
                            batch_gpu = batch_cpu.to(self.device, non_blocking=True)
                            
                            aug_batch = self.real_augmentor(batch_gpu)
                            inp_batch = self.preprocess(aug_batch)

                            try:
                                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                                    f = model.forward_features(inp_batch)
                            except Exception:
                                with autocast():
                                    f = model.forward_features(inp_batch)

                            f_cpu = f.detach().cpu().to(torch.float32)
                            
                            # 事前確保領域へコピー
                            final_bank_tensor[fill_pointer : fill_pointer + current_batch_size] = f_cpu
                            fill_pointer += current_batch_size
                            
                            pbar.update(current_batch_size)
                            
                            del batch_gpu, aug_batch, inp_batch, f
                            
            # 保存
            torch.save(final_bank_tensor, cache_path)
            logger.log(f"  Saved cache: {cache_path}")
            bank_list.append(final_bank_tensor)
            
            # ループ内でのempty_cacheは削除 (呼び出し元またはモデル切り替え時に実施)

        return bank_list


# ==========================================================
# 5. メイン最適化クラス
# ==========================================================
class MultiModelGM:
    def __init__(self, models, model_weights, target_class, args, device, initial_image=None):
        self.models = models
        self.model_weights = model_weights
        self.target_class = target_class
        self.args = args
        self.device = device

        self.generator = PyramidGenerator(
            target_size=args.image_size,
            start_size=args.pyramid_start_res,
            activation="sigmoid",
            initial_image=initial_image,
            noise_level=args.seed_noise_level,
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

        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        self.tv_loss_fn = TVLoss().to(device)

        # Pre-register normalization
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        # Pre-allocate for synthetic batch if possible? 
        # Variable size due to pyramid, but syn_augmentor output is fixed size.
        self.syn_aug_num = int(self.args.syn_aug)
    
    def _init_optimizer(self):
        return optim.Adam(self.generator.parameters(), lr=self.args.lr)

    def preprocess(self, img):
        return (img - self.norm_mean) / self.norm_std

    def get_grads_from_features(self, model, features, create_graph=False):
        params = list(model.classifier.parameters())
        logits = model.forward_head(features)
        
        # Targets recycling
        if not hasattr(self, "_target_labels") or self._target_labels.size(0) != features.size(0):
            self._target_labels = torch.tensor([self.target_class] * features.size(0), device=self.device)
            
        loss = F.cross_entropy(logits, self._target_labels)
        grads = torch.autograd.grad(loss, params, create_graph=create_graph)
        flat_grad = torch.cat([g.view(-1) for g in grads])
        return flat_grad

    def optimize_step(self, feature_bank_list_cpu):
        self.optimizer.zero_grad()
        syn_image = self.generator() # (1, 3, H, W)

        # --- Efficient Synthetic Batch Creation ---
        # リスト append -> cat を回避。先にTensor確保
        b, c, h, w = syn_image.shape
        # Augmentor出力は args.image_size (224) になる
        tgt_h, tgt_w = self.args.image_size, self.args.image_size
        
        # メモリ効率化: syn_batch を1つのTensorとして扱う
        # 毎回Augmentationを適用してスタック
        # syn_augmentorがTensorを受け取りTensorを返すと仮定
        syn_batch_tensor = torch.empty((self.syn_aug_num, c, tgt_h, tgt_w), device=self.device)
        
        for k in range(self.syn_aug_num):
            syn_batch_tensor[k] = self.syn_augmentor(syn_image).squeeze(0)
            
        inp_syn = self.preprocess(syn_batch_tensor)

        total_grad_loss = 0.0
        per_model_sims = {}

        for i, model in enumerate(self.models):
            weight = self.model_weights[i]
            if weight == 0:
                continue

            # Reset classifier weights (Need logic to optimize this frequency if priority C)
            model.reset_classifier()

            bank_cpu = feature_bank_list_cpu[i]
            
            # Random sampling from CPU bank
            idx_cpu = torch.randint(0, bank_cpu.size(0), (self.syn_aug_num,), device="cpu")
            real_feats = bank_cpu.index_select(0, idx_cpu).to(self.device, non_blocking=True)

            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    target_grad = self.get_grads_from_features(model, real_feats, create_graph=False)
                    syn_feats = model.forward_features(inp_syn)
                    syn_grad = self.get_grads_from_features(model, syn_feats, create_graph=True)
            except Exception:
                with autocast():
                    target_grad = self.get_grads_from_features(model, real_feats, create_graph=False)
                    syn_feats = model.forward_features(inp_syn)
                    syn_grad = self.get_grads_from_features(model, syn_feats, create_graph=True)

            sim = F.cosine_similarity(target_grad.unsqueeze(0).detach(), syn_grad.unsqueeze(0)).mean()
            loss_k = 1.0 - sim
            total_grad_loss += loss_k * weight

            model_name = self.args.encoder_names[i] if i < len(self.args.encoder_names) else f"model_{i}"
            per_model_sims[model_name] = sim.item()
            
            del real_feats # 明示的削除

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

    def run(self, feature_bank_list_cpu, save_dir, class_names, logger, global_pbar, gen_idx=0):
        # ... (変更なし、進捗表示ロジックのみ) ...
        logger.log(f"[{self.target_class}][Gen {gen_idx}] Optimization Start.")
        loss_history = []
        best_loss = float("inf")
        best_img_tensor = None

        local_pbar = tqdm(
            range(int(self.args.num_iterations)),
            desc=f"Exp {self.target_class}-G{gen_idx}",
            leave=False,
            position=1,
            dynamic_ncols=True,
        )

        for i in local_pbar:
            if i > 0 and i % int(self.args.pyramid_grow_interval) == 0:
                if self.generator.extend():
                    self.optimizer = self._init_optimizer()

            l_grad, l_tv, model_sims, current_total_loss = self.optimize_step(feature_bank_list_cpu)

            if current_total_loss < best_loss:
                best_loss = current_total_loss
                best_img_tensor = self.generator().detach().cpu()

            step_metrics = {"loss_grad": l_grad, "loss_tv": l_tv, "total_loss": current_total_loss}
            for m_name, m_sim in model_sims.items():
                step_metrics[f"sim_{m_name}"] = m_sim
            loss_history.append(step_metrics)

            local_pbar.set_description(f"G{gen_idx} L:{l_grad:.3f} R:{self.generator.levels[-1].shape[-1]}")
            global_pbar.update(1)

            if i % 500 == 0:
                with torch.no_grad():
                    save_image(
                        self.generator().detach().cpu(),
                        os.path.join(save_dir, f"step_{i:04d}_gen{gen_idx:02d}.png"),
                    )

        final_img = self.generator().detach().cpu()
        return final_img, best_img_tensor, {"loss_history": loss_history}