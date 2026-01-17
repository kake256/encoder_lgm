#!/bin/bash
# run_bank_2phase.sh
# 100GB VRAM向け: [Phase 1: キャッシュ作成] -> [Phase 2: "1つ見せ" 並列生成]
# 修正済み: 1プロセスだけ進捗を表示し、残りをバックグラウンドで実行する機能を追加
set -e

# ========================================================
# 設定エリア
# ========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
RESULTS_BASE_DIR="$PROJECT_DIR/classification_results"

EXPERIMENT_ROOT_DIR="$RESULTS_BASE_DIR/gene_experiment_feature_bank"
mkdir -p "$EXPERIMENT_ROOT_DIR"
export PYTHONPATH="$SRC_DIR:$PROJECT_DIR:$PYTHONPATH"

# 並列数 (100GB なら 8〜12 推奨)
# ※ 表示用の1プロセス + 裏方の(MAX-1)プロセス が動きます
MAX_JOBS_GEN=5

# データセット設定
DATASET_TYPE="imagenet-1k"
DATASET_SPLIT="train"
DATA_ROOT="$PROJECT_DIR/hf_datasets_cache"
mkdir -p "$DATA_ROOT"

# ターゲットクラス (スペース区切り)
# ★先頭のクラスが進捗表示用に使われます
TARGET_CLASSES="9 130 250 340 404 483 598 682 776 805 815 850 920 937 948 973 980 993 999"

# パラメータ (高品質設定)
USE_REAL=200
REAL_AUG=20
SYN_AUG=32
ITERATIONS=2500
NUM_GENERATIONS=1

MIN_SCALE=0.08
MAX_SCALE=1.0
NOISE_STD=0.2
NOISE_PROB=0.5
WEIGHT_TV=0.00025
PYRAMID_START_RES=16
PYRAMID_GROW_INTERVAL=400

# 文字攻撃設定
OVERLAY_TEXT=""
TEXT_COLOR="red"
FONT_SCALE=0.15
DISABLE_OCR_FLAG=""
if [ -z "$OVERLAY_TEXT" ]; then DISABLE_OCR_FLAG="--disable_ocr"; fi

# モデル設定
ENCODER_NAMES=(
  "facebook/dino-vitb16"
  "facebook/dinov2-base"
  "openai/clip-vit-base-patch16"
  "google/siglip-base-patch16-224"
)
PROJ_DIM_LIST=("0 0 0 0")
EXPERIMENTS=(
  "Only_V1:1.0,0.0,0.0,0.0"
  "Only_V2:0.0,1.0,0.0,0.0"
  "Only_CLIP:0.0,0.0,1.0,0.0"
  "Only_SigLIP:0.0,0.0,0.0,1.0"
  "Hybrid_V1_V2:1.0,1.0,0.0,0.0"
  "Hybrid_CLIP_SigLIP:0.0,0.0,1.0,1.0"
)

# ========================================================
# Phase 1: Feature Bank キャッシュの作成 (準備)
# ========================================================
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="$EXPERIMENT_ROOT_DIR/$TIMESTAMP"
CACHE_DIR="$EXPERIMENT_ROOT_DIR/feature_cache"
mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

echo "=========================================="
echo "Phase 1: Creating Feature Banks (Sequential)"
echo "=========================================="

CMD_PHASE1="cd \"$SRC_DIR\" && CUDA_VISIBLE_DEVICES=0 python3 main_feature_bank.py \
  --encoder_names ${ENCODER_NAMES[@]} \
  --projection_dims ${PROJ_DIM_LIST[0]} \
  --experiments ${EXPERIMENTS[@]} \
  --target_classes $TARGET_CLASSES \
  --dataset_type \"$DATASET_TYPE\" \
  --dataset_split \"$DATASET_SPLIT\" \
  --data_root \"$DATA_ROOT\" \
  --output_dir \"$BASE_OUTPUT_DIR\" \
  --cache_dir \"$CACHE_DIR\" \
  --use_real $USE_REAL \
  --real_aug $REAL_AUG \
  --min_scale $MIN_SCALE \
  --max_scale $MAX_SCALE \
  --noise_std $NOISE_STD \
  --noise_prob $NOISE_PROB \
  --overlay_text \"$OVERLAY_TEXT\" \
  --text_color \"$TEXT_COLOR\" \
  --font_scale $FONT_SCALE \
  --only_bank_creation \
  --log_file \"$BASE_OUTPUT_DIR/phase1_cache_log.txt\""

eval "$CMD_PHASE1"
echo "Phase 1 Completed."

# ========================================================
# Phase 2: 並列画像生成 (本番)
# ========================================================
echo "=========================================="
echo "Phase 2: High-Performance Generation"
echo "  - Monitor Class: The first one"
echo "  - Background Jobs: Parallel execution"
echo "=========================================="

export ENCODER_NAMES_STR="${ENCODER_NAMES[*]}"
export EXPERIMENTS_STR="${EXPERIMENTS[*]}"

# --- バックグラウンド実行用関数 (完全に静かに実行) ---
run_generation_silent() {
    local CLS_ID=$1
    local LOG_FILE="$BATCH_DIR/log_gen_class_${CLS_ID}.txt"
    # 画面には何も出さない
    CUDA_VISIBLE_DEVICES=0 python3 main_feature_bank.py \
      --encoder_names $ENCODER_NAMES_STR \
      --projection_dims $PROJ_DIMS \
      --experiments $EXPERIMENTS_STR \
      --target_classes $CLS_ID \
      --dataset_type "$DATASET_TYPE" \
      --dataset_split "$DATASET_SPLIT" \
      --data_root "$DATA_ROOT" \
      --output_dir "$BATCH_DIR" \
      --cache_dir "$CACHE_DIR" \
      --num_iterations $ITERATIONS \
      --num_generations $NUM_GENERATIONS \
      --use_real $USE_REAL \
      --real_aug $REAL_AUG \
      --syn_aug $SYN_AUG \
      --weight_tv $WEIGHT_TV \
      --overlay_text "$OVERLAY_TEXT" \
      --text_color "$TEXT_COLOR" \
      --font_scale $FONT_SCALE \
      --pyramid_start_res $PYRAMID_START_RES \
      --pyramid_grow_interval $PYRAMID_GROW_INTERVAL \
      --min_scale $MIN_SCALE \
      --max_scale $MAX_SCALE \
      --noise_std $NOISE_STD \
      --noise_prob $NOISE_PROB \
      $DISABLE_OCR_FLAG \
      --no_evaluation \
      --log_file "$LOG_FILE" > /dev/null 2>&1
}

# --- フォアグラウンド実行用関数 (進捗バーを表示) ---
run_generation_verbose() {
    local CLS_ID=$1
    local LOG_FILE="$BATCH_DIR/log_gen_class_${CLS_ID}.txt"
    echo ">>> Monitoring Progress for Class $CLS_ID <<<"
    # リダイレクトなしで実行してtqdmを表示させる
    CUDA_VISIBLE_DEVICES=0 python3 main_feature_bank.py \
      --encoder_names $ENCODER_NAMES_STR \
      --projection_dims $PROJ_DIMS \
      --experiments $EXPERIMENTS_STR \
      --target_classes $CLS_ID \
      --dataset_type "$DATASET_TYPE" \
      --dataset_split "$DATASET_SPLIT" \
      --data_root "$DATA_ROOT" \
      --output_dir "$BATCH_DIR" \
      --cache_dir "$CACHE_DIR" \
      --num_iterations $ITERATIONS \
      --num_generations $NUM_GENERATIONS \
      --use_real $USE_REAL \
      --real_aug $REAL_AUG \
      --syn_aug $SYN_AUG \
      --weight_tv $WEIGHT_TV \
      --overlay_text "$OVERLAY_TEXT" \
      --text_color "$TEXT_COLOR" \
      --font_scale $FONT_SCALE \
      --pyramid_start_res $PYRAMID_START_RES \
      --pyramid_grow_interval $PYRAMID_GROW_INTERVAL \
      --min_scale $MIN_SCALE \
      --max_scale $MAX_SCALE \
      --noise_std $NOISE_STD \
      --noise_prob $NOISE_PROB \
      $DISABLE_OCR_FLAG \
      --no_evaluation \
      --log_file "$LOG_FILE"
}

export -f run_generation_silent
export SRC_DIR DATASET_TYPE DATASET_SPLIT DATA_ROOT CACHE_DIR ITERATIONS NUM_GENERATIONS
export USE_REAL REAL_AUG SYN_AUG WEIGHT_TV OVERLAY_TEXT TEXT_COLOR FONT_SCALE
export PYRAMID_START_RES PYRAMID_GROW_INTERVAL MIN_SCALE MAX_SCALE NOISE_STD NOISE_PROB DISABLE_OCR_FLAG

# クラスリストを配列に変換
CLASS_ARRAY=($TARGET_CLASSES)
FIRST_CLASS=${CLASS_ARRAY[0]}
# 2番目以降のクラスを取得 (スライス)
REST_CLASSES=${CLASS_ARRAY[@]:1}

for PROJ_DIMS_VAL in "${PROJ_DIM_LIST[@]}"; do
  export PROJ_DIMS=$PROJ_DIMS_VAL
  export BATCH_DIR="$BASE_OUTPUT_DIR/gen_data"
  mkdir -p "$BATCH_DIR"
  
  # 1. 残りのクラスをバックグラウンドで一気に実行
  #    (xargsがプロセス管理をしてくれる)
  echo ">>> Starting background jobs for ${#CLASS_ARRAY[@]} classes..."
  if [ -n "$REST_CLASSES" ]; then
      echo "$REST_CLASSES" | tr ' ' '\n' | xargs -P $MAX_JOBS_GEN -I {} bash -c "cd $SRC_DIR && run_generation_silent {}" &
      BG_PID=$!
  else
      BG_PID=""
  fi

  # 2. 最初のクラスだけをフォアグラウンドで実行 (進捗バーが見える)
  cd "$SRC_DIR"
  run_generation_verbose $FIRST_CLASS
  
  # 3. フォアグラウンドが終わったら、バックグラウンドの完了を待つ
  if [ -n "$BG_PID" ]; then
      echo ">>> Monitor finished. Waiting for remaining background jobs..."
      wait $BG_PID
  fi
done

echo "All phases completed. Results at: $BASE_OUTPUT_DIR"