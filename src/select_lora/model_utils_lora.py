import math
import re
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from select_layer.model_utils_layer import (
        FeatureBankSystem,
        MultiModelGM as BaseMultiModelGM,
        EncoderClassifier as BaseEncoderClassifier,
        manage_model_allocation,
    )
except ModuleNotFoundError:
    import os
    import sys

    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _SRC_DIR = os.path.dirname(_THIS_DIR)
    _SELECT_LAYER_DIR = os.path.join(_SRC_DIR, "select_layer")
    if _SELECT_LAYER_DIR not in sys.path:
        sys.path.append(_SELECT_LAYER_DIR)

    from model_utils_layer import (
        FeatureBankSystem,
        MultiModelGM as BaseMultiModelGM,
        EncoderClassifier as BaseEncoderClassifier,
        manage_model_allocation,
    )


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear expects nn.Linear as base_layer")
        if rank <= 0:
            raise ValueError("rank must be > 0")

        self.base = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = self.base.in_features
        out_features = self.base.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        for p in self.base.parameters():
            p.requires_grad = False

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_out + (lora_out * self.scaling)


class EncoderClassifier(BaseEncoderClassifier):
    def __init__(
        self,
        encoder_model="openai/clip-vit-base-patch16",
        encoder=None,
        freeze_encoder=True,
        num_classes=1000,
        feature_source="pooler",
        projection_dim=2048,
        feature_layer=-1,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        lora_target_modules: Optional[List[str]] = None,
        lora_last_n_blocks=2,
    ):
        super().__init__(
            encoder_model=encoder_model,
            encoder=encoder,
            freeze_encoder=freeze_encoder,
            num_classes=num_classes,
            feature_source=feature_source,
            projection_dim=projection_dim,
            feature_layer=feature_layer,
        )

        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "v_proj", "query", "value", "qkv"]

        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_target_modules = [m.strip() for m in lora_target_modules if str(m).strip()]
        self.lora_last_n_blocks = int(lora_last_n_blocks)

        self._lora_modules: List[LoRALinear] = []
        self._lora_params: List[nn.Parameter] = []

        if self.lora_rank > 0:
            replaced = self._inject_lora_modules()
            if replaced == 0:
                print(
                    f"[Warning] LoRA: no target linear modules found for {self.encoder_model_name}. "
                    f"targets={self.lora_target_modules}"
                )

    def _extract_block_idx(self, module_name: str):
        patterns = [
            r"vision_model\.encoder\.layers\.(\d+)",
            r"encoder\.layers\.(\d+)",
            r"encoder\.layer\.(\d+)",
            r"blocks\.(\d+)",
            r"layer\.(\d+)",
            r"layers\.(\d+)",
        ]
        for p in patterns:
            m = re.search(p, module_name)
            if m:
                return int(m.group(1))
        return None

    def _matches_lora_target(self, module_name: str):
        leaf = module_name.split(".")[-1]
        return leaf in self.lora_target_modules

    def _replace_submodule(self, root: nn.Module, full_name: str, new_module: nn.Module):
        parts = full_name.split(".")
        parent = root
        for p in parts[:-1]:
            if p.isdigit():
                parent = parent[int(p)]
            else:
                parent = getattr(parent, p)

        leaf = parts[-1]
        if leaf.isdigit():
            parent[int(leaf)] = new_module
        else:
            setattr(parent, leaf, new_module)

    def _inject_lora_modules(self):
        candidates = []
        for name, module in self.encoder.named_modules():
            if isinstance(module, nn.Linear) and self._matches_lora_target(name):
                candidates.append((name, module, self._extract_block_idx(name)))

        if not candidates:
            return 0

        idxs = [idx for _, _, idx in candidates if idx is not None]
        min_idx = None
        if idxs and self.lora_last_n_blocks > 0:
            max_idx = max(idxs)
            min_idx = max_idx - self.lora_last_n_blocks + 1

        replaced = 0
        for name, module, idx in candidates:
            if min_idx is not None:
                if idx is None or idx < min_idx:
                    continue

            wrapped = LoRALinear(
                base_layer=module,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )
            self._replace_submodule(self.encoder, name, wrapped)
            self._lora_modules.append(wrapped)
            self._lora_params.extend([wrapped.lora_A, wrapped.lora_B])
            replaced += 1

        return replaced

    def has_lora(self):
        return len(self._lora_params) > 0

    def get_lora_parameters(self):
        return list(self._lora_params)

    def reset_lora_parameters(self):
        for m in self._lora_modules:
            m.reset_lora_parameters()


class MultiModelGM(BaseMultiModelGM):
    def __init__(self, models, model_weights, target_classes, args, device, initial_image=None):
        super().__init__(models, model_weights, target_classes, args, device, initial_image)
        self.grad_target = getattr(self.args, "grad_target", "lora")
        self.hybrid_lambda = float(getattr(self.args, "hybrid_lambda", 0.5))
        self.lora_reset_each_step = bool(getattr(self.args, "lora_reset_each_step", False))

        if self.grad_target not in {"classifier", "lora", "hybrid"}:
            raise ValueError(f"Unsupported grad_target={self.grad_target}")

    def _flatten_grads(self, grads, params):
        flat = []
        for g, p in zip(grads, params):
            if g is None:
                flat.append(torch.zeros_like(p).reshape(-1))
            else:
                flat.append(g.reshape(-1))
        return torch.cat(flat)

    def _grads_for_params(self, loss, params, create_graph=False, retain_graph=False):
        if len(params) == 0:
            raise ValueError("No parameters selected for gradient matching.")
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=True,
        )
        return self._flatten_grads(grads, params)

    def get_grads_from_features(self, model, features, tgt_class, create_graph=False):
        logits = model.forward_head(features)
        target_labels = torch.full((features.size(0),), int(tgt_class), device=self.device, dtype=torch.long)
        loss = F.cross_entropy(logits, target_labels)

        classifier_params = list(model.classifier.parameters())
        lora_params = model.get_lora_parameters() if hasattr(model, "get_lora_parameters") else []

        if self.grad_target == "classifier":
            return self._grads_for_params(
                loss,
                classifier_params,
                create_graph=create_graph,
                retain_graph=False,
            )

        if self.grad_target == "lora":
            if len(lora_params) == 0:
                raise ValueError(
                    "grad_target=lora but no LoRA parameters were found. "
                    "Set lora_rank>0 and ensure target modules match the model."
                )
            return self._grads_for_params(
                loss,
                lora_params,
                create_graph=create_graph,
                retain_graph=True,
            )

        # hybrid
        has_cls = len(classifier_params) > 0
        has_lora = len(lora_params) > 0
        if not has_cls and not has_lora:
            raise ValueError("hybrid grad target selected, but no classifier/LoRA params are available.")

        if has_cls:
            cls_grad = self._grads_for_params(
                loss,
                classifier_params,
                create_graph=create_graph,
                retain_graph=has_lora,
            )
        if has_lora:
            lora_grad = self._grads_for_params(
                loss,
                lora_params,
                create_graph=create_graph,
                retain_graph=True,
            )

        if has_cls and has_lora:
            w = min(max(self.hybrid_lambda, 0.0), 1.0)
            return torch.cat([(1.0 - w) * cls_grad, w * lora_grad])
        if has_cls:
            return cls_grad
        return lora_grad

    def optimize_step(self, real_image_pools_cpu):
        if self.lora_reset_each_step:
            for i, model in enumerate(self.models):
                if self.model_weights[i] == 0:
                    continue
                if hasattr(model, "reset_lora_parameters"):
                    model.reset_lora_parameters()

        return super().optimize_step(real_image_pools_cpu)
