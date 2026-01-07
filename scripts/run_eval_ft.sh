#!/bin/bash
set -euo pipefail

# ========================================================
# User config
# ========================================================
# Mixed Training Hyperparameters
# 実画像(5枚)と生成画像(50枚)のバランスを取るための重み
# 5.0 に設定すると、実画像が5倍の頻度でサンプリングされ、1:2 (Real:Syn) 程度の比率になる
MIXED_REAL_WEIGHT=5.0 

# Training settings
MIXED_EPOCHS=40
MIXED_BATCH_SIZE=32
MIXED_LR_BACKBONE=0.00001 # 1e-5
MIXED_LR_HEAD=0.001       # 1e-3
MIXED_WEIGHT_DECAY=0.0001

# ========================================================
# Path resolution
# ========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
# 使用するPythonスクリプトを指定 (Mixed Training版)
PYTHON_SCRIPT="$PROJECT_ROOT/src/evaluate_mixed_ft.py"

# Default synthetic dataset path
DEFAULT_DATASET_DIR="$PROJECT_ROOT/makeData/dataset_clean"

# Dataset selection defaults.
DEFAULT_DATASET_TYPE="imagenet"

# --------------------------------------------------------
# CUB Settings
# --------------------------------------------------------
DEFAULT_CUB_SOURCE="hf"
DEFAULT_CUB_HF_DATASET="Donghyun99/CUB-200-2011"
DEFAULT_CUB_HF_TRAIN_SPLIT="train"
DEFAULT_CUB_HF_TEST_SPLIT="test"
DEFAULT_CUB_REAL_DATA_DIR="$PROJECT_ROOT/data/CUB_200_2011"

# --------------------------------------------------------
# Food101 Settings
# --------------------------------------------------------
DEFAULT_FOOD_HF_DATASET="ethz/food101"
DEFAULT_FOOD_HF_TRAIN_SPLIT="train"
DEFAULT_FOOD_HF_TEST_SPLIT="validation"

# --------------------------------------------------------
# ImageNet Settings
# --------------------------------------------------------
DEFAULT_IMAGENET_HF_DATASET="imagenet-1k"
DEFAULT_IMAGENET_HF_TRAIN_SPLIT="train"
DEFAULT_IMAGENET_HF_TEST_SPLIT="validation"

# ========================================================
# Args parsing
# ========================================================
DATASET_DIR="$DEFAULT_DATASET_DIR"
DATASET_TYPE="$DEFAULT_DATASET_TYPE"

# CUB vars
CUB_SOURCE="$DEFAULT_CUB_SOURCE"
CUB_HF_DATASET="$DEFAULT_CUB_HF_DATASET"
CUB_HF_TRAIN_SPLIT="$DEFAULT_CUB_HF_TRAIN_SPLIT"
CUB_HF_TEST_SPLIT="$DEFAULT_CUB_HF_TEST_SPLIT"
REAL_DATA_DIR="$DEFAULT_CUB_REAL_DATA_DIR"

# Food101 vars
FOOD_HF_DATASET="$DEFAULT_FOOD_HF_DATASET"
FOOD_HF_TRAIN_SPLIT="$DEFAULT_FOOD_HF_TRAIN_SPLIT"
FOOD_HF_TEST_SPLIT="$DEFAULT_FOOD_HF_TEST_SPLIT"

# ImageNet vars
IMAGENET_HF_DATASET="$DEFAULT_IMAGENET_HF_DATASET"
IMAGENET_HF_TRAIN_SPLIT="$DEFAULT_IMAGENET_HF_TRAIN_SPLIT"
IMAGENET_HF_TEST_SPLIT="$DEFAULT_IMAGENET_HF_TEST_SPLIT"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    -t|--type)
      if [[ -n "${2:-}" && "$2" != -* ]]; then
        DATASET_TYPE="$2"
        shift
      else
        echo "Error: Argument for $1 is missing"
        exit 1
      fi
      ;;
    # --- CUB Args ---
    --cub_source)
      if [[ -n "${2:-}" && "$2" != -* ]]; then CUB_SOURCE="$2"; shift; fi ;;
    --cub_hf_dataset)
      if [[ -n "${2:-}" && "$2" != -* ]]; then CUB_HF_DATASET="$2"; shift; fi ;;
    --cub_hf_train_split)
      if [[ -n "${2:-}" && "$2" != -* ]]; then CUB_HF_TRAIN_SPLIT="$2"; shift; fi ;;
    --cub_hf_test_split)
      if [[ -n "${2:-}" && "$2" != -* ]]; then CUB_HF_TEST_SPLIT="$2"; shift; fi ;;
    -r|--real_dir)
      if [[ -n "${2:-}" && "$2" != -* ]]; then REAL_DATA_DIR="$2"; shift; fi ;;
    
    # --- Food101 Args ---
    --food_hf_dataset)
      if [[ -n "${2:-}" && "$2" != -* ]]; then FOOD_HF_DATASET="$2"; shift; fi ;;
    --food_hf_train_split)
      if [[ -n "${2:-}" && "$2" != -* ]]; then FOOD_HF_TRAIN_SPLIT="$2"; shift; fi ;;
    --food_hf_test_split)
      if [[ -n "${2:-}" && "$2" != -* ]]; then FOOD_HF_TEST_SPLIT="$2"; shift; fi ;;

    # --- ImageNet Args ---
    --imagenet_hf_dataset)
      if [[ -n "${2:-}" && "$2" != -* ]]; then IMAGENET_HF_DATASET="$2"; shift; fi ;;
    --imagenet_hf_train_split)
      if [[ -n "${2:-}" && "$2" != -* ]]; then IMAGENET_HF_TRAIN_SPLIT="$2"; shift; fi ;;
    --imagenet_hf_test_split)
      if [[ -n "${2:-}" && "$2" != -* ]]; then IMAGENET_HF_TEST_SPLIT="$2"; shift; fi ;;

    -*)
      # Ignore unknown args or implement help
      ;;
    *)
      DATASET_DIR="$1"
      ;;
  esac
  shift
done

# ========================================================
# Output dir & Config Generation
# ========================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_ROOT/mixed_ft_results/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

CONFIG_FILE="$OUTPUT_DIR/mixed_ft_config.json"
echo "{
  \"epochs\": $MIXED_EPOCHS,
  \"batch_size\": $MIXED_BATCH_SIZE,
  \"lr_backbone\": $MIXED_LR_BACKBONE,
  \"lr_head\": $MIXED_LR_HEAD,
  \"weight_decay\": $MIXED_WEIGHT_DECAY,
  \"real_weight\": $MIXED_REAL_WEIGHT
}" > "$CONFIG_FILE"

# ========================================================
# Preflight
# ========================================================
echo "========================================================"
echo " Mixed Training Evaluation Pipeline"
echo "========================================================"
echo " Project Root         : $PROJECT_ROOT"
echo " Python Script        : $PYTHON_SCRIPT"
echo " Syn Dataset Path     : $DATASET_DIR"
echo " Output Path          : $OUTPUT_DIR"
echo " Config File          : $CONFIG_FILE"
echo " --------------------------------------------------------"
echo " Dataset Type         : $DATASET_TYPE"
if [ "$DATASET_TYPE" = "cub" ]; then
  echo " CUB Source           : $CUB_SOURCE"
fi
echo "========================================================"

if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "Error: Python script not found at $PYTHON_SCRIPT"
  exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
  echo "Error: Synthetic dataset directory not found at $DATASET_DIR"
  exit 1
fi

# ========================================================
# Run
# ========================================================
echo ">>> Running Python Evaluation..."

if [ "$DATASET_TYPE" = "cub" ]; then
  python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_type "$DATASET_TYPE" \
    --cub_source "$CUB_SOURCE" \
    --cub_hf_dataset "$CUB_HF_DATASET" \
    --cub_hf_train_split "$CUB_HF_TRAIN_SPLIT" \
    --cub_hf_test_split "$CUB_HF_TEST_SPLIT" \
    --real_data_dir "$REAL_DATA_DIR" \
    --config "$CONFIG_FILE"

elif [ "$DATASET_TYPE" = "food101" ]; then
  python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_type "$DATASET_TYPE" \
    --food_hf_dataset "$FOOD_HF_DATASET" \
    --food_hf_train_split "$FOOD_HF_TRAIN_SPLIT" \
    --food_hf_test_split "$FOOD_HF_TEST_SPLIT" \
    --config "$CONFIG_FILE"

else
  # ImageNet execution
  python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_type "$DATASET_TYPE" \
    --imagenet_hf_dataset "$IMAGENET_HF_DATASET" \
    --imagenet_hf_train_split "$IMAGENET_HF_TRAIN_SPLIT" \
    --imagenet_hf_test_split "$IMAGENET_HF_TEST_SPLIT" \
    --config "$CONFIG_FILE"
fi

echo "========================================================"
echo " All Done."
echo " Results are saved in: $OUTPUT_DIR"
echo "========================================================"