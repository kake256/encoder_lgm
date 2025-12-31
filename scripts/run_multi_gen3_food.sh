#!/bin/bash
set -e

# ========================================================
# 0. 並列時のCPUスレッド制御（推奨）
# ========================================================
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# ========================================================
# 1. 環境設定
# ========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
RESULTS_BASE_DIR="$PROJECT_DIR/classification_results"

EXPERIMENT_ROOT_DIR="$RESULTS_BASE_DIR/gene_experiment_4models_comparison"
mkdir -p "$EXPERIMENT_ROOT_DIR"

export PYTHONPATH="$SRC_DIR:$PROJECT_DIR:$PYTHONPATH"

# ========================================================
# 2. データセット設定
# ========================================================
COMMON_DATA_ROOT="$PROJECT_DIR/data"

DATASET_TYPE="food101"
DATA_ROOT="$COMMON_DATA_ROOT"
TARGET_CLASSES="49 73 76 92"

# Food101 が 1-index 指定なら 1 にする
CLASS_ID_BASE=0
# CLASS_ID_BASE=1

# ========================================================
# 3. 実験パラメータ
# ========================================================
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
)

ITERATIONS=4000
AUGS_PER_STEP=64
NUM_GENERATIONS=1
LR=0.002
IMAGE_SIZE=224
NUM_REF_IMAGES=10

MIN_SCALE=0.08
MAX_SCALE=1.0
NOISE_STD=0.2
NOISE_PROB=0.5
WEIGHT_TV=0.00025
PYRAMID_START_RES=4

# ========================================================
# 4. 並列実行設定
# ========================================================
MAX_JOBS=4
NUM_GPUS=1

# ========================================================
# 5. 実験実行フェーズ
# ========================================================
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="$EXPERIMENT_ROOT_DIR/$TIMESTAMP"
mkdir -p "$BASE_OUTPUT_DIR"

BATCH_DIR="$BASE_OUTPUT_DIR/gen_data"
mkdir -p "$BATCH_DIR"

echo "=========================================="
echo "Dataset: $DATASET_TYPE"
echo "Classes: $TARGET_CLASSES"
echo "Output:  $BASE_OUTPUT_DIR"
echo "MAX_JOBS=$MAX_JOBS, NUM_GPUS=$NUM_GPUS"
echo "=========================================="

CLASS_LOOP=($TARGET_CLASSES)
PIDS=()
JOB_COUNTER=0

# ========================================================
# 6. クラス並列実行ループ
# ========================================================
for PROJ_DIMS in "${PROJ_DIM_LIST[@]}"; do
  for CLASS_ID in "${CLASS_LOOP[@]}"; do
    GPU_ID=$((JOB_COUNTER % NUM_GPUS))
    JOB_COUNTER=$((JOB_COUNTER + 1))

    LOG_FILE_PATH="$BATCH_DIR/class_${CLASS_ID}.log"

    echo "   [GPU $GPU_ID] Launching Class: $CLASS_ID"

    (
      # --- old ---
      # set -e

      # --- new ---
      # 個別プロセス失敗で全体を止めない
      set +e

      cd "$SRC_DIR" || exit 1

      CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_genimg.py \
        --encoder_names "${ENCODER_NAMES[@]}" \
        --projection_dims $PROJ_DIMS \
        --experiments "${EXPERIMENTS[@]}" \
        --target_classes "$CLASS_ID" \
        --class_id_base "$CLASS_ID_BASE" \
        --dataset_type "$DATASET_TYPE" \
        --data_root "$DATA_ROOT" \
        --output_dir "$BATCH_DIR" \
        --num_iterations "$ITERATIONS" \
        --lr "$LR" \
        --image_size "$IMAGE_SIZE" \
        --num_ref_images "$NUM_REF_IMAGES" \
        --augs_per_step "$AUGS_PER_STEP" \
        --weight_tv "$WEIGHT_TV" \
        --num_generations "$NUM_GENERATIONS" \
        --pyramid_start_res "$PYRAMID_START_RES" \
        --min_scale "$MIN_SCALE" \
        --max_scale "$MAX_SCALE" \
        --noise_std "$NOISE_STD" \
        --noise_prob "$NOISE_PROB" \
        --log_file "$LOG_FILE_PATH" \
        >> "$LOG_FILE_PATH" 2>&1

      exit_code=$?
      if [ $exit_code -ne 0 ]; then
        echo "[Runner] Process exited with code=$exit_code" >> "$LOG_FILE_PATH"
      fi
      exit $exit_code
    ) &

    PIDS+=($!)

    # 最大並列数に達したら待機
    if [ "${#PIDS[@]}" -ge "$MAX_JOBS" ]; then
      for pid in "${PIDS[@]}"; do
        wait "$pid" || true
      done
      PIDS=()
    fi
  done
done

# 残ジョブ待機
if [ "${#PIDS[@]}" -gt 0 ]; then
  for pid in "${PIDS[@]}"; do
    wait "$pid" || true
  done
fi

echo "=========================================="
echo "All Processes Finished."
echo "Results: $BASE_OUTPUT_DIR"
echo "=========================================="
