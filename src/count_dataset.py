import argparse
from pathlib import Path

# ==============================================================================
# 1. パス設定
# ==============================================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent 

# デフォルトの集計対象ディレクトリ
DEFAULT_TARGET_DIR = PROJECT_ROOT / "makeData" / "dataset_clean_ImageNet1k_200_typo"

# ==============================================================================
# 2. 集計ロジック
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="モデルごとにクラスごとの画像枚数を表示し、最後に合計を出力します")
    parser.add_argument("--target", "-t", type=str, default=None, help="集計対象のルートディレクトリ")
    args = parser.parse_args()

    # 対象ディレクトリの決定
    target_root = Path(args.target) if args.target else DEFAULT_TARGET_DIR

    print(f"\n==================================================")
    print(f" Dataset Counter (By Model)")
    print(f" Target: {target_root}")
    print(f"==================================================")

    if not target_root.exists():
        print(f"[Error] ディレクトリが見つかりません: {target_root}")
        return

    # --------------------------------------------------
    # データ収集
    # --------------------------------------------------
    # 構造: stats[model_name][class_name] = count
    stats = {}
    all_classes = set()

    class_dirs = [d for d in target_root.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print("[Warning] クラスフォルダが見つかりませんでした。")
        return

    for class_dir in class_dirs:
        class_name = class_dir.name
        all_classes.add(class_name)
        
        model_dirs = [d for d in class_dir.iterdir() if d.is_dir()]
        
        for model_dir in model_dirs:
            model_name = model_dir.name
            count = len(list(model_dir.glob("*.png")))
            
            if model_name not in stats:
                stats[model_name] = {}
            stats[model_name][class_name] = count

    # --------------------------------------------------
    # 表示 (モデルごとにブロックを分けて出力)
    # --------------------------------------------------
    sorted_models = sorted(stats.keys())
    sorted_classes = sorted(list(all_classes))
    
    # 最終的な合計を保存する辞書
    model_total_summary = {}

    if not sorted_models:
        print("[Result] 有効なモデルディレクトリが見つかりませんでした。")
        return

    for model in sorted_models:
        print(f"\n■ Model: {model}")
        print(f"-" * 45)
        
        model_total = 0
        
        for cls in sorted_classes:
            count = stats[model].get(cls, 0)
            
            # クラス名と枚数を表示
            print(f"  {cls:<35} : {count:>3}")
            
            model_total += count
            
        print(f"-" * 45)
        print(f"  {'[Subtotal]':<35} : {model_total:>3}")
        
        # 集計用に保存
        model_total_summary[model] = model_total

    # --------------------------------------------------
    # 最終集計 (Final Summary)
    # --------------------------------------------------
    print(f"\n\n==================================================")
    print(f" Final Summary (Total per Model)")
    print(f"==================================================")
    
    total_all_images = 0
    for model in sorted_models:
        count = model_total_summary[model]
        total_all_images += count
        print(f"  {model:<35} : {count:>4} images")
    
    print(f"--------------------------------------------------")
    print(f"  {'GRAND TOTAL':<35} : {total_all_images:>4} images")
    print(f"==================================================\n")

if __name__ == "__main__":
    main()