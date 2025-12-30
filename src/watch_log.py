import sys
import time
import os
from tqdm import tqdm

def main():
    if len(sys.argv) < 3:
        print("Usage: python watch_log.py <logfile> <iters_per_experiment>")
        sys.exit(1)

    log_file = sys.argv[1]
    
    # デフォルトの実験あたりの回数 (ログから合計値が取れなかった場合の予備)
    try:
        iters_per_exp = int(sys.argv[2])
    except ValueError:
        iters_per_exp = 4000 

    print(f"Waiting for log file: {log_file} ...")
    
    # ログファイルが生成されるまで待機
    timeout = 30
    start_wait = time.time()
    while not os.path.exists(log_file):
        time.sleep(0.5)
        if time.time() - start_wait > timeout:
            print("Timeout waiting for log file.")
            sys.exit(1)

    print(f"Watching: {log_file}")
    
    # プログレスバー初期化
    pbar = tqdm(total=iters_per_exp, unit="iter", dynamic_ncols=True, desc="Progress")
    
    last_log_step = 0      # ログ上の現在のステップ
    completed_exps = 0     # 完了した実験数
    total_steps_found = False # 合計ステップ数を見つけたか

    with open(log_file, 'r') as f:
        while True:
            line = f.readline()
            
            if not line:
                time.sleep(0.1)
                if "All Experiments Completed" in line:
                    break
                continue
            
            # 完了検知
            if "All Experiments Completed" in line:
                if pbar.total:
                    pbar.n = pbar.total
                    pbar.refresh()
                break

            # [連携] 合計ステップ数の自動取得
            if "__TOTAL_ESTIMATED_STEPS__" in line and not total_steps_found:
                try:
                    parts = line.strip().split()
                    real_total = int(parts[1])
                    
                    # バーの最大値を正しい合計値に変更
                    pbar.reset(total=real_total)
                    total_steps_found = True
                    
                    # 位置を復元
                    current_global_step = (completed_exps * iters_per_exp) + last_log_step
                    pbar.n = current_global_step
                    pbar.refresh()
                except:
                    pass

            # 進捗更新
            if "__PROGRESS__" in line:
                try:
                    parts = line.strip().split()
                    step_info = parts[1].split('/')
                    current_step = int(step_info[0])
                    loss_val = parts[2]
                    
                    # カウンタがリセットされた（次の実験へ）
                    if current_step < last_log_step:
                        completed_exps += 1
                        last_log_step = 0

                    # 現在の通算ステップ計算
                    global_current_step = (completed_exps * iters_per_exp) + current_step
                    
                    # バー更新
                    delta = global_current_step - pbar.n
                    if delta > 0:
                        pbar.update(delta)
                    
                    pbar.set_postfix(loss=loss_val)
                    last_log_step = current_step
                    
                except Exception:
                    continue

    pbar.close()
    print("Job (GPU 0) Completed.")

if __name__ == "__main__":
    main()