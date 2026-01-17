#!/bin/bash
set -euo pipefail

# ========================================================
# User Configuration
# ========================================================

# --- 1. Evaluator Models Selection ---
# 混同行列の類似度(Policy_3)を出すため、Generatorモデル(DINO, CLIP等)も含めることを推奨します。
#SELECTED_EVALUATORS="ResNet50 DINOv1 DINOv2 CLIP SigLIP OpenCLIP_ViT_B32 OpenCLIP_RN50 OpenCLIP_ConvNeXt MAE SwAV"
SELECTED_EVALUATORS="ResNet50 OpenCLIP_ViT_B32 OpenCLIP_RN50 OpenCLIP_ConvNeXt"

# --- 2. Augmentation & Visualization ---
ENABLE_AUGMENTATION="false"
# t-SNEの可視化 (メモリ不足やクラッシュ回避のため false 推奨)
ENABLE_TSNE="false"

# --- 3. Dataset Path ---
# Pythonスクリプトの場所 (適宜変更してください)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$PROJECT_ROOT/src/evaluate_linear.py"

# 合成データセットの場所
#DEFAULT_DATASET_DIR="$PROJECT_ROOT/makeData_midmid/dataset_clean"
DEFAULT_DATASET_DIR="$PROJECT_ROOT/makeData/dataset_clean"

# --- 4. Training Hyperparameters (Logistic Regression) ---
LOGREG_MAX_ITER=1000
LOGREG_C=1.0

# ========================================================
# Path resolution & Defaults
# ========================================================
DEFAULT_DATASET_TYPE="imagenet"

# CUB Settings
DEFAULT_CUB_SOURCE="hf"
DEFAULT_CUB_HF_DATASET="Donghyun99/CUB-200-2011"
DEFAULT_CUB_HF_TRAIN_SPLIT="train"
DEFAULT_CUB_HF_TEST_SPLIT="test"
DEFAULT_CUB_REAL_DATA_DIR="$PROJECT_ROOT/data/CUB_200_2011"

# Food101 Settings
DEFAULT_FOOD_HF_DATASET="ethz/food101"
DEFAULT_FOOD_HF_TRAIN_SPLIT="train"
DEFAULT_FOOD_HF_TEST_SPLIT="validation"

# ImageNet Settings
DEFAULT_IMAGENET_HF_DATASET="imagenet-1k"
DEFAULT_IMAGENET_HF_TRAIN_SPLIT="train"
DEFAULT_IMAGENET_HF_TEST_SPLIT="validation"

# ========================================================
# Internal Logic
# ========================================================

# Augmentation Flag
if [ "$ENABLE_AUGMENTATION" = "true" ]; then
  AUGMENT_FLAG="--augment"
  AUGMENT_STATUS="ON (Config)"
else
  AUGMENT_FLAG=""
  AUGMENT_STATUS="OFF"
fi

# TSNE Flag
if [ "$ENABLE_TSNE" = "false" ]; then
  TSNE_FLAG="--no_tsne"
  TSNE_STATUS="OFF (Skipped)"
else
  TSNE_FLAG=""
  TSNE_STATUS="ON"
fi

# Args Parsing
DATASET_DIR="$DEFAULT_DATASET_DIR"
DATASET_TYPE="$DEFAULT_DATASET_TYPE"
EVALUATORS="$SELECTED_EVALUATORS"

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

CONFIG_FILE=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    -a|--augment)
      AUGMENT_FLAG="--augment"
      AUGMENT_STATUS="ON (Argument)"
      ;;
    --no_tsne)
      TSNE_FLAG="--no_tsne"
      TSNE_STATUS="OFF (Argument)"
      ;;
    -t|--type)
      if [[ -n "${2:-}" && "$2" != -* ]]; then
        DATASET_TYPE="$2"
        shift
      else
        echo "Error: Argument for $1 is missing"
        exit 1
      fi
      ;;
    --evaluators)
      if [[ -n "${2:-}" && "$2" != -* ]]; then
        EVALUATORS="$2"
        shift
      else
        echo "Error: Argument for $1 is missing"
        exit 1
      fi
      ;;
    --config)
      if [[ -n "${2:-}" && "$2" != -* ]]; then CONFIG_FILE="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;
    --max_iter)
      if [[ -n "${2:-}" && "$2" != -* ]]; then LOGREG_MAX_ITER="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;
    --C)
      if [[ -n "${2:-}" && "$2" != -* ]]; then LOGREG_C="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;
    --cub_source)
      if [[ -n "${2:-}" && "$2" != -* ]]; then CUB_SOURCE="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;
    --cub_hf_dataset)
      if [[ -n "${2:-}" && "$2" != -* ]]; then CUB_HF_DATASET="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;
    --cub_hf_train_split)
      if [[ -n "${2:-}" && "$2" != -* ]]; then CUB_HF_TRAIN_SPLIT="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;
    --cub_hf_test_split)
      if [[ -n "${2:-}" && "$2" != -* ]]; then CUB_HF_TEST_SPLIT="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;
    -r|--real_dir)
      if [[ -n "${2:-}" && "$2" != -* ]]; then REAL_DATA_DIR="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;
    --food_hf_dataset)
      if [[ -n "${2:-}" && "$2" != -* ]]; then FOOD_HF_DATASET="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;
    --food_hf_train_split)
      if [[ -n "${2:-}" && "$2" != -* ]]; then FOOD_HF_TRAIN_SPLIT="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;
    --food_hf_test_split)
      if [[ -n "${2:-}" && "$2" != -* ]]; then FOOD_HF_TEST_SPLIT="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;
    --imagenet_hf_dataset)
      if [[ -n "${2:-}" && "$2" != -* ]]; then IMAGENET_HF_DATASET="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;
    --imagenet_hf_train_split)
      if [[ -n "${2:-}" && "$2" != -* ]]; then IMAGENET_HF_TRAIN_SPLIT="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;
    --imagenet_hf_test_split)
      if [[ -n "${2:-}" && "$2" != -* ]]; then IMAGENET_HF_TEST_SPLIT="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;
    -*)
      echo "Unknown option: $1"
      exit 1
      ;;
    *)
      DATASET_DIR="$1"
      ;;
  esac
  shift
done

# Output Dir
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_ROOT/evaluation_results_critique/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Generate Config
if [ -z "$CONFIG_FILE" ]; then
  CONFIG_FILE="$OUTPUT_DIR/probe_config.json"
  echo "{
    \"max_iter\": $LOGREG_MAX_ITER,
    \"C\": $LOGREG_C
  }" > "$CONFIG_FILE"
fi

# Prepare Evaluator Args
EVAL_ARGS=()
if [ -n "$EVALUATORS" ]; then
  EVAL_ARGS+=(--evaluators $EVALUATORS)
fi

echo "========================================================"
echo " Critical Evaluation Pipeline"
echo "========================================================"
echo " Project Root         : $PROJECT_ROOT"
echo " Python Script        : $PYTHON_SCRIPT"
echo " Syn Dataset Path     : $DATASET_DIR"
echo " Output Path          : $OUTPUT_DIR"
echo " Data Augmentation    : $AUGMENT_STATUS"
echo " t-SNE Visualization  : $TSNE_STATUS"
echo " Selected Evaluators  : ${EVALUATORS:-All (Default)}"
echo " Config File          : $CONFIG_FILE"
echo "========================================================"

if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "Error: Python script not found at $PYTHON_SCRIPT"
  exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
  echo "Error: Synthetic dataset directory not found at $DATASET_DIR"
  exit 1
fi

echo ">>> Installing Dependencies..."
# ↓ コメントアウトしました
# pip install transformers datasets scikit-learn pandas matplotlib sentencepiece tqdm pillow torchvision timm open_clip_torch --quiet

echo ">>> Running Python Evaluation..."

if [ "$DATASET_TYPE" = "cub" ]; then
  python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $AUGMENT_FLAG \
    $TSNE_FLAG \
    "${EVAL_ARGS[@]}" \
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
    $AUGMENT_FLAG \
    $TSNE_FLAG \
    "${EVAL_ARGS[@]}" \
    --dataset_type "$DATASET_TYPE" \
    --food_hf_dataset "$FOOD_HF_DATASET" \
    --food_hf_train_split "$FOOD_HF_TRAIN_SPLIT" \
    --food_hf_test_split "$FOOD_HF_TEST_SPLIT" \
    --config "$CONFIG_FILE"

else
  python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $AUGMENT_FLAG \
    $TSNE_FLAG \
    "${EVAL_ARGS[@]}" \
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

