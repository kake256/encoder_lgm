#!/bin/bash
# run_bank.sh
# Feature Bank対応 + ストリーミングモード + CPUキャッシュ管理版
# 目的: ImageNet全量(または一部)の特徴量をキャッシュし、マルチモデルで画像生成(蒸留)を行う
set -e

# ========================================================
# 1. 環境設定
# ========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
RESULTS_BASE_DIR="$PROJECT_DIR/classification_results"

# 結果保存ディレクトリ
EXPERIMENT_ROOT_DIR="$RESULTS_BASE_DIR/gene_experiment_feature_bank"
mkdir -p "$EXPERIMENT_ROOT_DIR"
export PYTHONPATH="$SRC_DIR:$PROJECT_DIR:$PYTHONPATH"

# 必要であればHugging Faceのキャッシュ先を指定 (容量確保のため)
# export HF_HOME="/path/to/large/disk/huggingface"

# ========================================================
# 2. データセット設定
# ========================================================
DATASET_TYPE="imagenet"
DATASET_SPLIT="train" # 'train'推奨 (約1300枚/クラス)。'validation'だと50枚/クラスしかありません。
DATA_ROOT=""          # ローカルにデータがある場合はパスを指定

TARGET_CLASSES="9 130 250 340 404 483 598 682 776 805 815 850 920 937 948 973 980 993 999"
#TARGET_CLASSES="9" 
#TARGET_CLASSES="815 850 920 937 948 973 980 993 999"

# ========================================================
# 3. Feature Bank & 最適化パラメータ (★重要設定)
# ========================================================
# [USE_REAL] 実画像の参照枚数 (Feature Bankの元ネタ)
#   0  : クラス内の全画像を使用 (推奨: 約1300枚。最も精度が高い)
#   N  : 先頭からN枚のみ使用 (例: 40)。設定を変えるとキャッシュの再計算が発生します。
USE_REAL=200

# [REAL_AUG] 実画像1枚あたりの拡張数 (Feature Bankのサイズ決定)
#   例: 1300枚 x 20倍 = 26,000個の特徴量をキャッシュ
#   10〜20程度が推奨。多いほどメモリを使うが分布が密になる。
REAL_AUG=20

# [SYN_AUG] 1ステップあたりの比較枚数 (バッチサイズ)
#   生成画像を何通りに拡張して比較するか。
#   Bankからランダムにこの数だけ特徴量を取り出して比較する。
SYN_AUG=64

# [ITERATIONS] 最適化のループ回数
#   キャッシュ作成後はここが実行時間の支配要因になる。
ITERATIONS=2500

# --- Augmentation (データ拡張) の強度設定 ---
MIN_SCALE=0.08      # RandomResizedCropの最小スケール (小さいほど局所を見る)
MAX_SCALE=1.0       # RandomResizedCropの最大スケール
NOISE_STD=0.2       # ガウシアンノイズの標準偏差
NOISE_PROB=0.5      # ノイズを加える確率

# --- その他の学習パラメータ ---
WEIGHT_TV=0.00025       # TV Loss (滑らかさ) の重み

# --- ピラミッド解像度制御 (画像を細かくしていくペース) ---
PYRAMID_START_RES=16      # 開始解像度 (16x16)
PYRAMID_GROW_INTERVAL=400 # 何イテレーションごとに解像度を上げるか (論文設定: 200)

# --- 文字攻撃 (Text Overlay) 設定 ---
# 空文字 "" なら適用なし。指定すると実画像に文字を合成してから特徴量を計算する。
OVERLAY_TEXT=""
TEXT_COLOR="red"
FONT_SCALE=0.15

# --- ジョブ制御 ---
MAX_JOBS=1     # 同時に走らせるプロセス数 (VRAM容量に合わせて調整)
NUM_GPUS=1     # 使用するGPUの総数 (0番から順に使われる)

# ========================================================
# 4. モデル設定
# ========================================================
# 使用するモデルのリスト (Hugging Face ID)
ENCODER_NAMES=(
  "facebook/dino-vitb16"           # Index 0: DINOv1
  "facebook/dinov2-base"           # Index 1: DINOv2
  "openai/clip-vit-base-patch16"   # Index 2: CLIP
  "google/siglip-base-patch16-224" # Index 3: SigLIP
)

# プロジェクション層の次元 (基本は0=なし)
PROJ_DIM_LIST=("0 0 0 0")

# 実験設定リスト
# 書式: "実験名:重み0,重み1,重み2,重み3"
# ※重みの並び順は ENCODER_NAMES のインデックスに対応
EXPERIMENTS=(
  # --- 単体モデル (基準用) ---
  "Only_V1:1.0,0.0,0.0,0.0"       # DINOv1のみ
  "Only_V2:0.0,1.0,0.0,0.0"       # DINOv2のみ
  "Only_CLIP:0.0,0.0,1.0,0.0"     # CLIPのみ
  "Only_SigLIP:0.0,0.0,0.0,1.0"   # SigLIPのみ

  # --- マルチモデル (2モデルの組み合わせ) ---
  # 重みを1.0ずつにすると単純加算。比率を変えたい場合は 0.5, 0.5 などに調整。
  
  # 1. 同系統の組み合わせ
  "Hybrid_V1_V2:1.0,1.0,0.0,0.0"         # DINO同士
  "Hybrid_CLIP_SigLIP:0.0,0.0,1.0,1.0"   # 言語系同士

  # 2. DINOv1 とのハイブリッド
  "Hybrid_V1_CLIP:1.0,0.0,1.0,0.0"
  "Hybrid_V1_SigLIP:1.0,0.0,0.0,1.0"

  # 3. DINOv2 とのハイブリッド
  "Hybrid_V2_CLIP:0.0,1.0,1.0,0.0"
  "Hybrid_V2_SigLIP:0.0,1.0,0.0,1.0"
)

# ========================================================
# 5. 実行ループ (変更不要)
# ========================================================
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="$EXPERIMENT_ROOT_DIR/$TIMESTAMP"
CACHE_DIR="$EXPERIMENT_ROOT_DIR/cache" # 特徴量キャッシュの保存先

mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

echo "=========================================="
echo "Starting Feature Bank Optimization"
echo "  - Use Real Images: $USE_REAL (0=All)"
echo "  - Cache Augmentations: $REAL_AUG x"
echo "  - Step Batch Size: $SYN_AUG"
echo "  - Cache Dir: $CACHE_DIR"
echo "  - Output Dir: $BASE_OUTPUT_DIR"
echo "=========================================="

for PROJ_DIMS in "${PROJ_DIM_LIST[@]}"; do
  BATCH_DIR="$BASE_OUTPUT_DIR/gen_data"
  mkdir -p "$BATCH_DIR"

  CLASS_LOOP=($TARGET_CLASSES)

  PIDS=()
  JOB_COUNTER=0

  for CLASS_ID in "${CLASS_LOOP[@]}"; do
    # GPU割り当ての計算 (ラウンドロビン)
    GPU_ID=$((JOB_COUNTER % NUM_GPUS))
    JOB_COUNTER=$((JOB_COUNTER + 1))

    LOG_FILENAME="class_${CLASS_ID}.log"
    LOG_FILE_PATH="$BATCH_DIR/$LOG_FILENAME"

    # Pythonスクリプトの実行
    # ★修正点: 進捗表示のためバックグラウンド実行(&)と出力破棄(> /dev/null)を削除しました
    CMD="cd \"$SRC_DIR\" && CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_feature_bank.py \
      --encoder_names ${ENCODER_NAMES[@]} \
      --projection_dims $PROJ_DIMS \
      --experiments ${EXPERIMENTS[@]} \
      --target_classes $CLASS_ID \
      --dataset_type \"$DATASET_TYPE\" \
      --dataset_split \"$DATASET_SPLIT\" \
      --data_root \"$DATA_ROOT\" \
      --output_dir \"$BATCH_DIR\" \
      --cache_dir \"$CACHE_DIR\" \
      --num_iterations $ITERATIONS \
      --use_real $USE_REAL \
      --real_aug $REAL_AUG \
      --syn_aug $SYN_AUG \
      --weight_tv $WEIGHT_TV \
      --overlay_text \"$OVERLAY_TEXT\" \
      --text_color \"$TEXT_COLOR\" \
      --font_scale $FONT_SCALE \
      --pyramid_start_res $PYRAMID_START_RES \
      --pyramid_grow_interval $PYRAMID_GROW_INTERVAL \
      --min_scale $MIN_SCALE \
      --max_scale $MAX_SCALE \
      --noise_std $NOISE_STD \
      --noise_prob $NOISE_PROB \
      --streaming \
      --log_file \"$LOG_FILE_PATH\""

    echo "Running Class $CLASS_ID on GPU $GPU_ID..."
    eval "$CMD"
    
    # 直列実行になるため wait は不要ですが、念のため
    # PIDS+=($!)
  done
done

echo "Done. Results at: $BASE_OUTPUT_DIR"