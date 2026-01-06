# =========================
# optimized_model_utils.py  (旧・安定版, 特徴保持なし)
# =========================
import os
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
    print("Warning, 'timm' library not found. DINOv3 or timm-based models cannot be loaded automatically.")


def manage_model_allocation(models, weights, device):
    for i, model in enumerate(models):
        if weights[i] > 0:
            model.to(device)
        else:
            model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class RandomGaussianNoise(nn.Module):
    def __init__(self, mean=0.0, std=1.0, p=0.5):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, images):
        if torch.rand(1).item() >= self.p:
            return images
        noise = torch.randn_like(images) * self.std + self.mean
        return images + noise


class TVLoss(nn.Module):
    def forward(self, img):
        b, c, h, w = img.size()
        h_tv = torch.pow((img[:, :, 1:, :] - img[:, :, :h-1, :]), 2).mean()
        w_tv = torch.pow((img[:, :, :, 1:] - img[:, :, :, :w-1]), 2).mean()
        return h_tv + w_tv


class EncoderClassifier(nn.Module):
    def __init__(
        self,
        encoder_model="openai/clip-vit-base-patch16",
        encoder=None,
        freeze_encoder=True,
        num_classes=1000,
        feature_source="pooler",
        projection_dim=2048
    ):
        super().__init__()
        self.encoder_model_name = encoder_model
        self.use_timm_encoder = False

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
            elif "dino" in self.encoder_model_name and "v3" not in self.encoder_model_name and "v2" not in self.encoder_model_name and "timm" not in self.encoder_model_name:
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


class PyramidGenerator(nn.Module):
    def __init__(self, target_size=224, start_size=16, activation="sigmoid", initial_image=None, noise_level=0.0):
        super().__init__()
        self.target_size = target_size
        self.activation = activation

        color_correlation_svd_sqrt = torch.tensor([
            [0.26, 0.09, 0.02],
            [0.27, 0.00, -0.05],
            [0.27, -0.09, 0.03]
        ])
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
                initial_image, size=(start_size, start_size),
                mode="bilinear", align_corners=False, antialias=True
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
                level_tensor, size=(self.target_size, self.target_size),
                mode="bilinear", align_corners=False, antialias=True
            )
            image = image + upsampled
        image = self.linear_decorrelate_color(image)
        if self.activation == "sigmoid":
            return torch.sigmoid(2 * image)
        return image


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
            noise_level=args.seed_noise_level
        ).to(device)

        self.optimizer = self._init_optimizer()
        self.augmentor = T.Compose([
            T.RandomResizedCrop(args.image_size, scale=(args.min_scale, args.max_scale), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            T.ColorJitter(0.1, 0.1, 0.1, 0.05),
            RandomGaussianNoise(mean=0.0, std=args.noise_std, p=args.noise_prob),
        ])

        self.scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
        self.tv_loss_fn = TVLoss().to(device)

        # 旧版, experimentごとに毎回計算して保持する.
        self.cached_real_features = [None] * len(models)

    def _init_optimizer(self):
        return optim.Adam(self.generator.parameters(), lr=self.args.lr)

    def preprocess(self, img):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        return (img - mean) / std

    def get_grads_from_features(self, model, features, create_graph=False):
        params = list(model.classifier.parameters())
        logits = model.forward_head(features)
        targets = torch.tensor([self.target_class] * features.size(0), device=self.device)
        loss = F.cross_entropy(logits, targets)
        grads = torch.autograd.grad(loss, params, create_graph=create_graph)
        flat_grad = torch.cat([g.view(-1) for g in grads])
        return flat_grad

    def precompute_real_features(self, real_images_pool):
        indices = torch.randperm(len(real_images_pool))[:min(self.args.num_ref_images, len(real_images_pool))]
        real_batch = real_images_pool[indices].detach()

        aug_real = self.augmentor(real_batch)
        inp_real = self.preprocess(aug_real)

        for i, model in enumerate(self.models):
            if self.model_weights[i] == 0:
                continue

            with torch.no_grad():
                try:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        feats = model.forward_features(inp_real)
                except Exception:
                    with autocast():
                        feats = model.forward_features(inp_real)

            self.cached_real_features[i] = feats.detach()

    def optimize_step(self, _unused):
        self.optimizer.zero_grad()
        syn_image = self.generator()

        syn_batch_list = []
        for _ in range(self.args.augs_per_step):
            syn_batch_list.append(self.augmentor(syn_image))
        syn_batch = torch.cat(syn_batch_list, dim=0)
        inp_syn = self.preprocess(syn_batch)

        total_grad_loss = 0.0
        per_model_sims = {}

        for i, model in enumerate(self.models):
            weight = self.model_weights[i]
            if weight == 0:
                continue

            model.reset_classifier()
            real_feats = self.cached_real_features[i]
            if real_feats is None:
                continue

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

        loss_tv = self.tv_loss_fn(syn_image)
        total_loss = (total_grad_loss * self.args.weight_grad) + (loss_tv * self.args.weight_tv)

        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        return float(total_grad_loss.item()), float(loss_tv.item()), per_model_sims, float(total_loss.item())

    def run(self, real_images_pool, save_dir, class_names, logger, global_pbar, gen_idx=0):
        logger.log(f"[{self.target_class}][Gen {gen_idx}] Optimization Start.")
        loss_history = []
        best_loss = float("inf")
        best_img_tensor = None

        local_pbar = tqdm(
            range(self.args.num_iterations),
            desc=f"Exp {self.target_class}-G{gen_idx}",
            leave=False,
            position=1,
            dynamic_ncols=True
        )

        for i in local_pbar:
            if i > 0 and i % self.args.pyramid_grow_interval == 0:
                if self.generator.extend():
                    self.optimizer = self._init_optimizer()

            l_grad, l_tv, model_sims, current_total_loss = self.optimize_step(real_images_pool)

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
                    save_image(self.generator().detach().cpu(),
                               os.path.join(save_dir, f"step_{i:04d}_gen{gen_idx:02d}.png"))

        final_img = self.generator().detach().cpu()
        return final_img, best_img_tensor, {"loss_history": loss_history}
