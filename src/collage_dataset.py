import argparse
from pathlib import Path
from PIL import Image
import math

# ==============================================================================
# 1. パス設定
# ==============================================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent 

DEFAULT_TARGET_DIR = PROJECT_ROOT / "makeData" / "dataset_clean_ImageNet1k_200"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "outputs" / "collages" / "ImageNet1k_200"

# ==============================================================================
# 2. ロジック
# ==============================================================================

def create_collage(model_name, image_paths, save_dir):
    if not image_paths:
        return

    images = [Image.open(p) for p in image_paths]
    img_width, img_height = images[0].size

    num_images = len(images)
    #cols = math.ceil(math.sqrt(num_images))
    cols = 4
    rows = math.ceil(num_images / cols)

    collage_width = cols * img_width
    collage_height = rows * img_height
    collage_image = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))

    for index, img in enumerate(images):
        x = (index % cols) * img_width
        y = (index // cols) * img_height
        collage_image.paste(img, (x, y))

    save_path = save_dir / f"collage_{model_name}.png"
    collage_image.save(save_path)
    print(f"  [Success] Saved: {save_path.name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", "-t", type=str, default=None)
    parser.add_argument("--out", "-o", type=str, default=None)
    args = parser.parse_args()

    target_root = Path(args.target) if args.target else DEFAULT_TARGET_DIR
    save_root = Path(args.out) if args.out else DEFAULT_SAVE_DIR

    if not target_root.exists():
        print(f"[Error] Directory not found: {target_root}")
        return

    save_root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # データ収集 (数値順にソート)
    # --------------------------------------------------
    model_images = {}
    
    # ディレクトリ名の先頭の数字を数値として評価してソート
    try:
        class_dirs = sorted(
            [d for d in target_root.iterdir() if d.is_dir()],
            key=lambda d: int(d.name.split('_')[0])
        )
    except ValueError:
        # もし数値から始まらないフォルダが混ざっていた場合のフォールバック
        print("[Info] Numerical sort failed for some folders, falling back to lexicographical sort.")
        class_dirs = sorted([d for d in target_root.iterdir() if d.is_dir()])
    
    for class_dir in class_dirs:
        model_dirs = [d for d in class_dir.iterdir() if d.is_dir()]
        
        for model_dir in model_dirs:
            model_name = model_dir.name
            
            # 画像ファイル名もソートして最初の1枚を取得
            images = sorted(list(model_dir.glob("*.png")))
            if images:
                if model_name not in model_images:
                    model_images[model_name] = []
                model_images[model_name].append(images[0])

    # --------------------------------------------------
    # 実行
    # --------------------------------------------------
    for model_name in sorted(model_images.keys()):
        print(f"Generating collage for Model: {model_name}...")
        create_collage(model_name, model_images[model_name], save_root)

if __name__ == "__main__":
    main()