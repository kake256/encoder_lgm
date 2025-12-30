import torch
import torch.nn as nn
from transformers import CLIPVisionModel, Dinov2Model, ViTModel, SiglipVisionModel

# [追加] 最新モデル(DINOv3等)用にtimmを利用
try:
    import timm
except ImportError:
    timm = None
    print("Warning: 'timm' library not found. DINOv3 or timm-based models cannot be loaded automatically.")

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
        
        # --- エンコーダーの準備 ---
        if encoder is not None:
            self.encoder = encoder
            if not hasattr(encoder, 'config'): 
                self.use_timm_encoder = True
        else:
            # モデル名に応じたロード分岐
            if "siglip" in self.encoder_model_name:
                self.encoder = SiglipVisionModel.from_pretrained(encoder_model)
                
            elif self.encoder_model_name.startswith("openai/clip") or "laion" in self.encoder_model_name:
                self.encoder = CLIPVisionModel.from_pretrained(encoder_model)
                
            elif "dinov2" in self.encoder_model_name and "timm" not in self.encoder_model_name:
                # HuggingFaceのDINOv2
                self.encoder = Dinov2Model.from_pretrained(encoder_model)
                
            elif "dino" in self.encoder_model_name and "v3" not in self.encoder_model_name and "v2" not in self.encoder_model_name and "timm" not in self.encoder_model_name:
                # DINOv1 (facebook/dino-...)
                self.encoder = ViTModel.from_pretrained(encoder_model)
                
            elif "dinov3" in self.encoder_model_name or self.encoder_model_name.startswith("timm/"):
                # [DINOv3 / timm対応]
                if timm is None:
                    raise ImportError("timm library is required for DINOv3 or timm models.")
                
                print(f"Loading via timm: {encoder_model}")
                # "timm/"プレフィックスを除去してロード
                model_name = encoder_model.replace("timm/", "")
                
                # num_classes=0 でHeadを除去し特徴量のみ出力する設定
                self.encoder = timm.create_model(
                    model_name, 
                    pretrained=True, 
                    num_classes=0
                )
                self.use_timm_encoder = True
            else:
                # フォールバック
                try:
                    self.encoder = ViTModel.from_pretrained(encoder_model)
                except:
                    raise ValueError(f"Unsupported encoder model: {encoder_model}")

        # --- feature_source と embed_dim の調整 ---
        if self.use_timm_encoder:
            self.embed_dim = self.encoder.num_features
            self.feature_source = feature_source 
        else:
            # Transformersモデル用設定
            if "siglip" in self.encoder_model_name:
                if feature_source == 'pooler': self.feature_source = 'mean'
                else: self.feature_source = feature_source
            elif self.encoder_model_name.startswith("openai/clip"):
                if feature_source not in ['pooler', 'cls', 'mean']:
                    self.feature_source = 'pooler'
                else:
                    self.feature_source = feature_source
            elif "dinov2" in self.encoder_model_name:
                if feature_source == 'pooler': self.feature_source = 'cls'
                else: self.feature_source = feature_source
            elif "dino" in self.encoder_model_name:
                if feature_source == 'pooler' and not hasattr(self.encoder.config, 'pooler_type'):
                     self.feature_source = 'cls'
                else: self.feature_source = feature_source
            else:
                self.feature_source = feature_source

            cfg = self.encoder.config
            self.embed_dim = cfg.hidden_size

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # プロジェクション層
        if projection_dim > 0:
            self.projector = nn.Sequential(
                nn.Linear(self.embed_dim, projection_dim),
                nn.ReLU(inplace=True)
            )
            classifier_in_dim = projection_dim
        else:
            self.projector = nn.Identity()
            classifier_in_dim = self.embed_dim

        # 分類器
        self.classifier = nn.Linear(classifier_in_dim, num_classes)
        self.reset_classifier()

    def reset_classifier(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

    def extract_features(self, outputs):
        """モデルの出力からベクトルを抽出"""
        
        # [timm対応] Tensorが直接返ってきた場合
        if isinstance(outputs, torch.Tensor):
            return outputs

        # Transformers対応
        if "siglip" in self.encoder_model_name:
            if self.feature_source == 'pooler' and hasattr(outputs, 'pooler_output'):
                return outputs.pooler_output
            elif self.feature_source == 'mean':
                return outputs.last_hidden_state.mean(dim=1)
            else: 
                return outputs.last_hidden_state[:, 0, :]

        elif self.encoder_model_name.startswith("openai/clip"):
            if self.feature_source == 'pooler': return outputs.pooler_output
            elif self.feature_source == 'cls': return outputs.last_hidden_state[:, 0, :]
            elif self.feature_source == 'mean': return outputs.last_hidden_state[:, 1:, :].mean(dim=1)
            
        elif "dino" in self.encoder_model_name:
            if self.feature_source == 'cls': return outputs.last_hidden_state[:, 0, :]
            elif self.feature_source == 'mean': return outputs.last_hidden_state[:, 1:, :].mean(dim=1)
        
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, x):
        if self.use_timm_encoder:
            # timmはTensorを直接受け取る
            outputs = self.encoder(x)
        else:
            # Transformersはキーワード引数
            outputs = self.encoder(pixel_values=x, output_hidden_states=False)
            
        features = self.extract_features(outputs)
        projected_features = self.projector(features)
        logits = self.classifier(projected_features)
        return logits

# --- Helper Modules (変更なし) ---
class TVLoss(nn.Module):
    def forward(self, img):
        b, c, h, w = img.size()
        h_tv = torch.pow((img[:, :, 1:, :] - img[:, :, :h-1, :]), 2).mean()
        w_tv = torch.pow((img[:, :, :, 1:] - img[:, :, :, :w-1]), 2).mean()
        return h_tv + w_tv

class PyramidGenerator(nn.Module):
    def __init__(self, target_size=224, start_size=16, activation='sigmoid', initial_image=None, noise_level=0.0):
        super().__init__()
        self.target_size = target_size
        self.activation = activation
        
        color_correlation_svd_sqrt = torch.tensor([
            [0.26, 0.09, 0.02],
            [0.27, 0.00, -0.05],
            [0.27, -0.09, 0.03]
        ])
        self.register_buffer('color_correlation', color_correlation_svd_sqrt)
        max_norm = torch.max(torch.linalg.norm(color_correlation_svd_sqrt, dim=0))
        self.register_buffer('max_norm', max_norm)
        
        normalized_matrix = color_correlation_svd_sqrt / max_norm
        try:
            inverse_matrix = torch.linalg.inv(normalized_matrix)
        except RuntimeError:
            inverse_matrix = torch.linalg.pinv(normalized_matrix)
        self.register_buffer('inverse_color_correlation', inverse_matrix)

        if initial_image is not None:
            init_low_res = torch.nn.functional.interpolate(initial_image, size=(start_size, start_size), mode='bilinear', align_corners=False, antialias=True)
            eps = 1e-6
            if activation == 'sigmoid':
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
            self.levels = nn.ParameterList([
                nn.Parameter(torch.randn(1, 3, start_size, start_size) * 0.1)
            ])

    def extend(self):
        current_res = max([p.shape[-1] for p in self.levels])
        if current_res >= self.target_size: return False 
        new_res = min(current_res * 2, self.target_size)
        old_len = len(self.levels)
        new_len = old_len + 1
        with torch.no_grad():
            for p in self.levels: p.mul_(old_len / new_len)
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
            upsampled = torch.nn.functional.interpolate(level_tensor, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False, antialias=True)
            image = image + upsampled
        image = self.linear_decorrelate_color(image)
        if self.activation == 'sigmoid': return torch.sigmoid(2 * image) 
        return image