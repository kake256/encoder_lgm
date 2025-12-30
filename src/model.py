import torch
import torch.nn as nn
from transformers import CLIPVisionModel, Dinov2Model, ViTModel

# ------------------------------
# Encoder + Projection + Linear Classifier
# ------------------------------
class EncoderClassifier(nn.Module):
    def __init__(
        self,
        encoder_model="openai/clip-vit-base-patch16",
        encoder=None,  # [追加] 事前ロード済みのエンコーダを受け取る引数
        freeze_encoder=True,
        num_classes=1000, 
        feature_source="pooler",
        projection_dim=2048  # 0の場合はプロジェクションなし
    ):
        super().__init__()
        self.encoder_model_name = encoder_model
        self.freeze_encoder = freeze_encoder
        self.feature_source = feature_source

        # --- エンコーダーの準備 ---
        if encoder is not None:
            # [追加] 外部から渡されたエンコーダを使用（高速化・メモリ節約）
            self.encoder = encoder
        else:
            # 従来どおり文字列からロード
            if self.encoder_model_name.startswith("openai/clip"):
                self.encoder = CLIPVisionModel.from_pretrained(encoder_model)
            elif self.encoder_model_name.startswith("facebook/dinov2"):
                self.encoder = Dinov2Model.from_pretrained(encoder_model)
            elif self.encoder_model_name.startswith("facebook/dino"):
                self.encoder = ViTModel.from_pretrained(encoder_model)
            else:
                raise ValueError(f"Unsupported encoder model: {encoder_model}")

        # --- 特徴抽出ソースの検証と調整 ---
        # エンコーダがロードされた後にチェックを行う（共通処理）
        if self.encoder_model_name.startswith("openai/clip"):
            if self.feature_source not in ['pooler', 'cls', 'mean']:
                print(f"Warning: Invalid feature_source '{self.feature_source}' for CLIP. Defaulting to 'pooler'.")
                self.feature_source = 'pooler'
        
        elif self.encoder_model_name.startswith("facebook/dinov2"):
            if self.feature_source == 'pooler':
                print(f"Warning: 'pooler' not available for DINOv2. Defaulting to 'cls'.")
                self.feature_source = 'cls'

        elif self.encoder_model_name.startswith("facebook/dino"):
            if self.feature_source == 'pooler' and not hasattr(self.encoder.config, 'pooler_type'):
                 self.feature_source = 'cls'

        cfg = self.encoder.config
        self.embed_dim = cfg.hidden_size

        if freeze_encoder:
            # エンコーダのパラメータを固定
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            print("Encoder parameters are NOT frozen (fine-tuning mode).")

        # プロジェクション層の有無を切り替え
        # Projectorは特徴抽出器の一部とみなし、固定パラメータとして扱います（勾配計算対象外）
        if projection_dim > 0:
            # print(f"Projection Layer Enabled: {self.embed_dim} -> {projection_dim}")
            self.projector = nn.Sequential(
                nn.Linear(self.embed_dim, projection_dim),
                nn.ReLU(inplace=True)
            )
            classifier_in_dim = projection_dim
        else:
            # print("Projection Layer Disabled (Using Raw Features)")
            self.projector = nn.Identity()
            classifier_in_dim = self.embed_dim

        # 分類器
        self.classifier = nn.Linear(classifier_in_dim, num_classes)
        
        # 初期化
        self.reset_classifier()

    def reset_classifier(self):
        """分類器の重みをランダムに再初期化する (LGMの要件)"""
        # Linear Gradient Matchingでは、毎回ランダムな重みWに対して勾配をマッチさせます
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

    def extract_features(self, outputs):
        """エンコーダの出力から特徴量を抽出する"""
        if self.encoder_model_name.startswith("openai/clip"):
            if self.feature_source == 'pooler':
                return outputs.pooler_output
            elif self.feature_source == 'cls':
                return outputs.last_hidden_state[:, 0, :]
            elif self.feature_source == 'mean':
                return outputs.last_hidden_state[:, 1:, :].mean(dim=1)
        
        elif self.encoder_model_name.startswith("facebook/dino"):
            if self.feature_source == 'cls':
                return outputs.last_hidden_state[:, 0, :]
            elif self.feature_source == 'mean':
                return outputs.last_hidden_state[:, 1:, :].mean(dim=1)
            return outputs.last_hidden_state[:, 0, :]
        
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        else:
            return outputs.last_hidden_state[:, 0, :]

    def forward(self, x):
        pixel_values = x
        outputs = self.encoder(pixel_values=pixel_values, output_hidden_states=False)
        features = self.extract_features(outputs)
        
        # 射影層を通す (Identityの場合はそのまま通る)
        projected_features = self.projector(features)
        
        # 分類器に通す
        logits = self.classifier(projected_features)
        return logits