#!/bin/bash
set -euo pipefail

# ========================================================
# User config, augmentation default.
# ========================================================
ENABLE_AUGMENTATION="true"

# ========================================================
# Path resolution.
# ========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$PROJECT_ROOT/src/evaluate_linear.py"

# Default synthetic dataset path (ImageNet default)
DEFAULT_DATASET_DIR="$PROJECT_ROOT/makeData/dataset_clean"
#DEFAULT_DATASET_DIR="$PROJECT_ROOT/makeData/dataset_clean_food101"

# Dataset selection defaults.
DEFAULT_DATASET_TYPE="imagenet"
#DEFAULT_DATASET_TYPE="food101"

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
# ImageNet Settings (New)
# --------------------------------------------------------
DEFAULT_IMAGENET_HF_DATASET="imagenet-1k"
DEFAULT_IMAGENET_HF_TRAIN_SPLIT="train"
DEFAULT_IMAGENET_HF_TEST_SPLIT="validation"

# ========================================================
# Apply augmentation default.
# ========================================================
if [ "$ENABLE_AUGMENTATION" = "true" ]; then
  AUGMENT_FLAG="--augment"
  AUGMENT_STATUS="ON (Config)"
else
  AUGMENT_FLAG=""
  AUGMENT_STATUS="OFF"
fi

# ========================================================
# Args parsing.
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
    -a|--augment)
      AUGMENT_FLAG="--augment"
      AUGMENT_STATUS="ON (Argument)"
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
    # --- CUB Args ---
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
    
    # --- Food101 Args ---
    --food_hf_dataset)
      if [[ -n "${2:-}" && "$2" != -* ]]; then FOOD_HF_DATASET="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;
    --food_hf_train_split)
      if [[ -n "${2:-}" && "$2" != -* ]]; then FOOD_HF_TRAIN_SPLIT="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;
    --food_hf_test_split)
      if [[ -n "${2:-}" && "$2" != -* ]]; then FOOD_HF_TEST_SPLIT="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;

    # --- ImageNet Args ---
    --imagenet_hf_dataset)
      if [[ -n "${2:-}" && "$2" != -* ]]; then IMAGENET_HF_DATASET="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;
    --imagenet_hf_train_split)
      if [[ -n "${2:-}" && "$2" != -* ]]; then IMAGENET_HF_TRAIN_SPLIT="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;
    --imagenet_hf_test_split)
      if [[ -n "${2:-}" && "$2" != -* ]]; then IMAGENET_HF_TEST_SPLIT="$2"; shift; else echo "Error: Arg missing for $1"; exit 1; fi ;;

    -*)
      echo "Unknown option: $1"
      echo "Usage:"
      echo "  bash run_eval_linear.sh [dataset_path] [-a|--augment] [--type|-t imagenet|cub|food101]"
      echo "    For CUB(HF):    [--cub_source hf] [--cub_hf_dataset ID] ..."
      echo "    For Food101:    [--food_hf_dataset ID] [--food_hf_train_split train] [--food_hf_test_split validation]"
      echo "    For ImageNet:   [--imagenet_hf_dataset ID] [--imagenet_hf_train_split train] [--imagenet_hf_test_split validation]"
      exit 1
      ;;
    *)
      DATASET_DIR="$1"
      ;;
  esac
  shift
done

# ========================================================
# Output dir.
# ========================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_ROOT/evaluation_results_critique/${TIMESTAMP}"

# ========================================================
# Preflight.
# ========================================================
echo "========================================================"
echo " Critical Evaluation Pipeline"
echo "========================================================"
echo " Project Root         : $PROJECT_ROOT"
echo " Python Script        : $PYTHON_SCRIPT"
echo " Syn Dataset Path     : $DATASET_DIR"
echo " Output Path          : $OUTPUT_DIR"
echo " Data Augmentation    : $AUGMENT_STATUS"
echo " --------------------------------------------------------"
echo " Dataset Type         : $DATASET_TYPE"

if [ "$DATASET_TYPE" = "cub" ]; then
  echo " CUB Source           : $CUB_SOURCE"
  if [ "$CUB_SOURCE" = "hf" ]; then
    echo " CUB HF Dataset       : $CUB_HF_DATASET"
  else
    echo " CUB Local Real Dir   : $REAL_DATA_DIR"
  fi
elif [ "$DATASET_TYPE" = "food101" ]; then
  echo " Food HF Dataset      : $FOOD_HF_DATASET"
  echo " Food HF Train Split  : $FOOD_HF_TRAIN_SPLIT"
  echo " Food HF Test Split   : $FOOD_HF_TEST_SPLIT"
elif [ "$DATASET_TYPE" = "imagenet" ]; then
  echo " ImageNet HF Dataset  : $IMAGENET_HF_DATASET"
  echo " ImageNet Train Split : $IMAGENET_HF_TRAIN_SPLIT"
  echo " ImageNet Test Split  : $IMAGENET_HF_TEST_SPLIT"
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
# Run.
# ========================================================
echo ">>> Installing Dependencies (if needed)..."
pip install transformers datasets scikit-learn pandas matplotlib sentencepiece tqdm pillow torchvision --quiet

echo ">>> Running Python Evaluation..."
mkdir -p "$OUTPUT_DIR"

if [ "$DATASET_TYPE" = "cub" ]; then
  python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $AUGMENT_FLAG \
    --dataset_type "$DATASET_TYPE" \
    --cub_source "$CUB_SOURCE" \
    --cub_hf_dataset "$CUB_HF_DATASET" \
    --cub_hf_train_split "$CUB_HF_TRAIN_SPLIT" \
    --cub_hf_test_split "$CUB_HF_TEST_SPLIT" \
    --real_data_dir "$REAL_DATA_DIR"

elif [ "$DATASET_TYPE" = "food101" ]; then
  # Food101 execution
  python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $AUGMENT_FLAG \
    --dataset_type "$DATASET_TYPE" \
    --food_hf_dataset "$FOOD_HF_DATASET" \
    --food_hf_train_split "$FOOD_HF_TRAIN_SPLIT" \
    --food_hf_test_split "$FOOD_HF_TEST_SPLIT"

else
  # ImageNet execution
  python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $AUGMENT_FLAG \
    --dataset_type "$DATASET_TYPE" \
    --imagenet_hf_dataset "$IMAGENET_HF_DATASET" \
    --imagenet_hf_train_split "$IMAGENET_HF_TRAIN_SPLIT" \
    --imagenet_hf_test_split "$IMAGENET_HF_TEST_SPLIT"
fi

echo "========================================================"
echo " All Done."
echo " Results are saved in: $OUTPUT_DIR"
echo "========================================================"