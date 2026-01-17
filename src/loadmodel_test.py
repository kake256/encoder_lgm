# check_models_loading.py
# 内容: evaluate_linear2_config.py で定義された全モデルのロードおよび推論テスト

import os
import sys
import torch
import numpy as np

# プロジェクト内のモジュールを読み込む
try:
    from evaluate_linear2_config import AVAILABLE_EVAL_MODELS, DEVICE
    from evaluate_linear2_models import FeatureExtractor
except ImportError as e:
    print("Error: 設定ファイルまたはモデル定義ファイルが見つかりません。")
    print("このスクリプトを 'evaluate_linear2.py' と同じディレクトリに置いて実行してください。")
    print(f"詳細: {e}")
    sys.exit(1)

def check_library_dependencies():
    print("=" * 60)
    print(" 1. Dependency Check")
    print("=" * 60)
    
    # timm check
    try:
        import timm
        print(f" [OK] timm installed (ver: {timm.__version__})")
    except ImportError:
        print(" [NG] timm is NOT installed. Please run: pip install timm")

    # open_clip check
    try:
        import open_clip
        print(f" [OK] open_clip installed (ver: {open_clip.__version__})")
    except ImportError:
        print(" [NG] open_clip is NOT installed. Please run: pip install open_clip_torch")
        
    print("-" * 60)

def test_model_loading():
    print("\n" + "=" * 60)
    print(" 2. Model Loading & Forward Pass Test")
    print("=" * 60)
    print(f" Device: {DEVICE}")
    
    # ダミー入力 (Batch=1, RGB, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    
    # 全モデルをループ
    for name, model_id in AVAILABLE_EVAL_MODELS.items():
        print(f"\nTarget: {name}")
        print(f"  ID: {model_id}")
        
        # -------------------------------------------------
        # Test 1: Load Pretrained
        # -------------------------------------------------
        print("  [1] Pretrained Loading...", end=" ")
        try:
            # pretrained=True でロード
            extractor = FeatureExtractor(model_identifier=model_id, pretrained=True)
            
            # パラメータ数計算
            num_params = sum(p.numel() for p in extractor.core.parameters()) / 1e6
            print(f"OK (Params: {num_params:.2f}M)")
            
            # Forward check
            with torch.no_grad():
                # 内部構造によっては forward ではなく encode_image などを呼ぶ必要があるが
                # FeatureExtractor.extract_features のロジックの一部を簡易再現
                if extractor.type == "open_clip":
                    out = extractor.core.encode_image(dummy_input)
                elif extractor.type == "swav":
                    out = extractor.core(dummy_input)
                elif extractor.type == "timm":
                    out = extractor.core(dummy_input)
                else:
                    # HF models
                    res = extractor.core(pixel_values=dummy_input)
                    if hasattr(res, "pooler_output") and res.pooler_output is not None:
                        out = res.pooler_output
                    elif hasattr(res, "last_hidden_state"):
                        out = res.last_hidden_state[:, 0]
                    else:
                        out = res[0]
                
                feat_dim = out.shape[-1]
                print(f"      Forward Check: OK (Output Dim: {feat_dim})")
                
            # メモリ解放
            del extractor
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"FAILED")
            print(f"      Error: {e}")
            # エラーが出ても次のモデルへ進む

        # -------------------------------------------------
        # Test 2: Load Scratch (Random Init)
        # -------------------------------------------------
        print("  [2] Scratch Loading...   ", end=" ")
        try:
            # pretrained=False でロード
            extractor = FeatureExtractor(model_identifier=model_id, pretrained=False)
            
            # 念のためパラメータ数が同じか確認
            num_params_scratch = sum(p.numel() for p in extractor.core.parameters()) / 1e6
            print(f"OK (Params: {num_params_scratch:.2f}M)")
            
            del extractor
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"FAILED")
            print(f"      Error: {e}")

if __name__ == "__main__":
    check_library_dependencies()
    test_model_loading()
    print("\nDone.")