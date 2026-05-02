import os
import shutil

def organize_all_models(input_root, output_base):
    """
    input_root: /encoder_lgm/ImageNet-100 (この下にモデル名のフォルダがある)
    output_base: /encoder_lgm/makeData/dataset_clean_imagenet100
    """
    
    # 1. 指定されたルート配下のフォルダ（モデル名）を取得
    if not os.path.exists(input_root):
        print(f"エラー: 入力ディレクトリ {input_root} が存在しません。")
        return

    # モデル名のフォルダ（CLIP, SigLIPなど）をループ
    for model_name in os.listdir(input_root):
        model_dir = os.path.join(input_root, model_name)
        
        # ディレクトリでない場合はスキップ
        if not os.path.isdir(model_dir):
            continue
            
        print(f"--- 処理中: モデル名 [{model_name}] ---")

        # 2. 各モデルフォルダ内のファイルをループ
        for filename in os.listdir(model_dir):
            if filename.endswith(".png"):
                # 元のファイル名からカテゴリ名を抽出 (例: 0_Bonnet -> 0_bonnet)
                name_without_ext = os.path.splitext(filename)[0]
                category_name = name_without_ext.lower()

                # 3. 新しいファイル名と保存先パスの作成
                # 例: category_name_result_gen00.png
                new_filename = f"{category_name}_result_gen00.png"
                
                # 構造: 出力先 / カテゴリ名 / モデル名 / ファイル名
                target_dir = os.path.join(output_base, category_name, model_name)
                target_file_path = os.path.join(target_dir, new_filename)

                # 4. 実行（フォルダ作成と移動）
                os.makedirs(target_dir, exist_ok=True)
                source_file_path = os.path.join(model_dir, filename)

                try:
                    # コピーではなく移動(shutil.move)を使用
                    shutil.move(source_file_path, target_file_path)
                    print(f"  [移動済] {filename} -> {category_name}/{model_name}/")
                except Exception as e:
                    print(f"  [エラー] {filename}: {e}")

if __name__ == "__main__":
    # --- ここでパスを指定してください ---
    input_directory = "../Food-101"  # 元データがある場所
    output_directory = "../makeData/dataset_clean_food101_orignal"  # 出力したい場所
    
    organize_all_models(input_directory, output_directory)
    print("\nすべての処理が完了しました。")