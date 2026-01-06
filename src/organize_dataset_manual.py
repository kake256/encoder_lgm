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

# ★変更点: 親フォルダではなく、対象の日付フォルダを直接指定します
# (自動検出による別フォルダの誤検知を防ぐため)
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "classification_results" / "gene_experiment_4models_comparison" / "20260101_192533"

# ★変更点: Food101用の出力先に設定
OUTPUT_DATASET_DIR = PROJECT_ROOT / "makeData" / "dataset_clean_food101"

# 収集対象のキーワード
TARGET_KEYWORDS = ["result_gen"]

# 除外ファイル
EXCLUDE_KEYWORDS = ["ref_pool", ".json", ".txt", ".log"]

# ==============================================================================
# 2. ロジック部
# ==============================================================================

def get_latest_timestamp_dir(root_dir):
    """(今回は使用しませんが、ロジックとして残します) 最新の日付フォルダを取得"""
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
        # ★変更点: 設定されたパス直下に gen_data があれば、それを直接採用する (自動検出をスキップ)
        if (DEFAULT_RESULTS_ROOT / "gen_data").exists():
            input_path = DEFAULT_RESULTS_ROOT
            print(f"[Info] 指定されたディレクトリを直接使用します: {input_path}")
        else:
            # 指定パスに gen_data がない場合のみ、子フォルダから最新を探す (旧ロジック)
            print(f"[Info] 指定パス直下に gen_data が見つからないため、最新のサブフォルダを検索します: {DEFAULT_RESULTS_ROOT}")
            input_path = get_latest_timestamp_dir(DEFAULT_RESULTS_ROOT)
            
        if input_path is None:
            print(f"[Error] 実験フォルダが見つかりません: {DEFAULT_RESULTS_ROOT}")
            return

    # 2. gen_data 階層への移動
    if (input_path / "gen_data").exists():
        data_root = input_path / "gen_data"
    else:
        # gen_dataフォルダがない場合（直下にモデルフォルダがある場合など）への対応
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
    
    if not exp_dirs:
        print(f"[Warning] ソースフォルダ内にディレクトリが見つかりません。パスを確認してください。")
    
    total_count = 0
    # ★追加: クラスごとの枚数を集計する辞書
    class_counts = {}

    pbar = tqdm(exp_dirs, desc="Processing Experiments")
    
    for exp_dir in pbar:
        # モデル名/実験名 (例: food101_Hybrid_CLIP_SigLIP)
        model_name = exp_dir.name
        
        # クラスフォルダ (例: 9_breakfast_burrito...)
        class_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
        
        if not class_dirs:
            # クラスフォルダが見つからない場合のデバッグ表示
            # tqdmの表示崩れを防ぐため write を使用
            pbar.write(f"[Info] {model_name} 内にクラスフォルダが見つかりません (空の可能性があります)")
            continue

        for class_dir in class_dirs:
            class_name = class_dir.name 
            
            # 保存先パス: Output / クラス名 / モデル名
            dest_dir = OUTPUT_DATASET_DIR / class_name / model_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # 直下のPNGのみ取得
            files = list(class_dir.glob("*.png"))
            
            for img_file in files:
                if is_target_file(img_file.name):
                    
                    # 新しいファイル名を作成: モデル名 + "_" + 元のファイル名
                    new_filename = f"{model_name}_{img_file.name}"
                    
                    # 重複チェック
                    final_name = get_unique_filename(dest_dir, new_filename)
                    
                    try:
                        shutil.copy2(img_file, dest_dir / final_name)
                        total_count += 1
                        
                        # ★追加: クラスごとのカウントを更新
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        
                    except Exception as e:
                        print(f"Copy Error: {e}")

    print(f"\n[Success] 完了しました。")
    print(f"合計 {total_count} 枚の画像をコピーしました。")
    print(f"保存先: {OUTPUT_DATASET_DIR}")
    
    # ★追加: クラス別内訳の表示
    if class_counts:
        print("\n=== クラス別コピー枚数 ===")
        # クラス名順にソートして表示
        for cls_name, count in sorted(class_counts.items()):
            print(f" {cls_name:<30} : {count}枚")
        print("==========================")

if __name__ == "__main__":
    main()