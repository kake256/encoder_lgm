import shutil
from pathlib import Path

def restructure_directories():
    # 1. 元のベースとなるディレクトリのパス
    # ※ご自身の環境に合わせて適宜変更してください
    base_path_str = "../makeData/imagenet100_FULL_prototype"
    base_dir = Path(base_path_str)
    
    # 【移動スイッチ】Trueで実際の移動を実行します
    EXECUTE_MOVE = True

    if not base_dir.exists():
        print(f"エラー: 指定されたディレクトリが見つかりません - {base_dir}")
        return

    # 2. ベースディレクトリ内の「モデル」フォルダ（例: dataset_nearest_CLIP）をループ
    for model_dir in base_dir.iterdir():
        if model_dir.is_dir():
            
            # 3. フォルダ名からサフィックス（例: CLIP）を抽出
            prefix = "dataset_nearest_"
            if model_dir.name.startswith(prefix):
                # "dataset_nearest_" を取り除いた部分を取得
                model_suffix = model_dir.name[len(prefix):] 
            else:
                # 万が一 prefix がない場合は、アンダースコアの最後の部分などを取得する安全策
                model_suffix = model_dir.name.split("_")[-1]

            # 4. 新しい配置先のベースディレクトリを作成
            # 例: imagenet100_FULL_prototype_CLIP
            new_base_dir_name = f"{base_dir.name}_{model_suffix}"
            new_base_dir = base_dir.parent / new_base_dir_name
            
            if EXECUTE_MOVE:
                # 新しいベースディレクトリを作成（すでに存在する場合は何もしない）
                new_base_dir.mkdir(parents=True, exist_ok=True)
                
                # 5. モデルフォルダの中のインデックスフォルダ（例: 0_bonnet）を移動
                for index_dir in model_dir.iterdir():
                    if index_dir.is_dir():
                        new_target_dir = new_base_dir / index_dir.name
                        
                        # 競合（すでに移動先に同じ名前がある場合）を回避
                        if new_target_dir.exists():
                            print(f"スキップ: 移動先にすでにフォルダが存在します - {new_target_dir}")
                        else:
                            shutil.move(str(index_dir), str(new_target_dir))
                            print(f"移動完了: {index_dir.name} -> {new_base_dir.name}/")
                
                # 6. 中身が空になったモデルフォルダをお掃除
                try:
                    model_dir.rmdir()
                    print(f"お掃除完了: 空になった {model_dir.name} フォルダを削除しました\n")
                except OSError:
                    print(f"注意: 他のファイルが残っているため {model_dir.name} は削除しませんでした\n")

    # （任意）もし元の base_dir 自体も完全に空になったら削除する処理
    if EXECUTE_MOVE:
        try:
            base_dir.rmdir()
            print(f"完了: 元のベースディレクトリ {base_dir.name} も空になったため削除しました")
        except OSError:
            pass

    print("すべての処理が完了しました！")

if __name__ == "__main__":
    restructure_directories()