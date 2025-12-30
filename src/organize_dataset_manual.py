import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# 1. パスとキーワードの設定
# ==============================================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent 

# 入力元ルート
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "classification_results" / "gene_experiment_4models_fast"

# 出力先ルート
# この下に「Class_Name / Model_Name / Image」が作られます
OUTPUT_DATASET_DIR = PROJECT_ROOT / "makeData" / "dataset_clean"
# OUTPUT_DATASET_DIR = PROJECT_ROOT / "makeData" / "dataset_text_overlay"

# 収集対象のキーワード
TARGET_KEYWORDS = ["result_gen"]

# 除外ファイル
EXCLUDE_KEYWORDS = ["ref_pool", ".json", ".txt", ".log"]

# ==============================================================================
# 2. ロジック部
# ==============================================================================

def get_latest_timestamp_dir(root_dir):
    """最新の日付フォルダを取得"""
    if not root_dir.exists():
        return None
    dirs = [d for d in root_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not dirs:
        return None
    return sorted(dirs, key=lambda x: x.name)[-1]

def get_unique_filename(dest_dir, filename):
    """同名ファイルの重複回避"""
    if not (dest_dir / filename).exists():
        return filename
    stem, ext = os.path.splitext(filename)
    counter = 1
    while (dest_dir / f"{stem}_{counter}{ext}").exists():
        counter += 1
    return f"{stem}_{counter}{ext}"

def is_target_file(filename):
    """対象のpngファイルか判定"""
    if not filename.endswith(".png"): return False
    if any(k in filename for k in EXCLUDE_KEYWORDS): return False
    
    for k in TARGET_KEYWORDS:
        if k in filename:
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="実験結果をクラス＞モデル順に整理します")
    parser.add_argument("--input", "-i", type=str, default=None, help="手動でパスを指定する場合")
    parser.add_argument("--clean", action="store_true", help="実行前に出力先を空にする")
    args = parser.parse_args()

    # 1. 入力ソースの特定
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = get_latest_timestamp_dir(DEFAULT_RESULTS_ROOT)
        if input_path is None:
            print(f"[Error] 実験フォルダが見つかりません: {DEFAULT_RESULTS_ROOT}")
            return

    # 2. gen_data 階層への移動
    if (input_path / "gen_data").exists():
        data_root = input_path / "gen_data"
    else:
        data_root = input_path

    print(f"==================================================")
    print(f" Dataset Organizer (Class > Model > Image)")
    print(f"==================================================")
    print(f" Source : {data_root}")
    print(f" Target : {OUTPUT_DATASET_DIR}")
    print(f"==================================================")

    if not data_root.exists():
        print(f"[Error] ソースパスが存在しません: {data_root}")
        return

    # 3. 出力先の準備
    if args.clean and OUTPUT_DATASET_DIR.exists():
        print("Cleaning old dataset...")
        shutil.rmtree(OUTPUT_DATASET_DIR)
    
    # 4. 走査とコピー実行
    exp_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    total_count = 0
    
    pbar = tqdm(exp_dirs, desc="Processing Experiments")
    
    for exp_dir in pbar:
        # モデル名/実験名 (例: Only_V1)
        model_name = exp_dir.name
        
        # クラスフォルダ (例: 9_ostrich...)
        class_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
        
        for class_dir in class_dirs:
            class_name = class_dir.name 
            
            # 【ここを変更しました】
            # 保存先パス: Output / クラス名 / モデル名
            dest_dir = OUTPUT_DATASET_DIR / class_name / model_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # 直下のPNGのみ取得
            files = list(class_dir.glob("*.png"))
            
            for img_file in files:
                if is_target_file(img_file.name):
                    # ファイル名はそのまま（フォルダで分かれているため）
                    final_name = get_unique_filename(dest_dir, img_file.name)
                    
                    try:
                        shutil.copy2(img_file, dest_dir / final_name)
                        total_count += 1
                    except Exception as e:
                        print(f"Copy Error: {e}")

    print(f"\n[Success] 完了しました。")
    print(f"合計 {total_count} 枚の画像をコピーしました。")
    print(f"保存先: {OUTPUT_DATASET_DIR}")

if __name__ == "__main__":
    main()