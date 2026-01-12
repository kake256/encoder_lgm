#!/bin/bash
# run_eval_sequential.sh
set -euo pipefail

# ========================================================
# User Configuration
# ========================================================

# Options: linear_lbfgs, linear_torch, partial_ft, lora, full_ft
# Note: The loop below will override EVAL_MODE to run lbfgs then torch.
#EVAL_MODE="linear_lbfgs"
#EVAL_MODE="linear_torch" 
#EVAL_MODE="partial_ft"
#EVAL_MODE="lora"
#EVAL_MODE="full_ft"

# Evaluators (space-separated names, must match keys in AVAILABLE_EVAL_MODELS)
#SELECTED_EVALUATORS="ResNet50 OpenCLIP_ViT_B32 OpenCLIP_RN50 OpenCLIP_ConvNeXt"
#SELECTED_EVALUATORS="ResNet50 DINOv1 DINOv2 CLIP SigLIP MAE SwAV OpenCLIP_RN50 OpenCLIP_ViT_B32"
SELECTED_EVALUATORS="ResNet50 MAE SwAV" #"OpenCLIP_RN50 OpenCLIP_ViT_B32"

#ENABLE_AUGMENTATION="false"
ENABLE_AUGMENTATION="true"
ENABLE_TSNE="false"  # kept for compatibility, not used

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Update this to point to your latest python script
PYTHON_SCRIPT="$PROJECT_ROOT/src/evaluate_linear2.py"

DEFAULT_DATASET_DIR="$PROJECT_ROOT/makeData/dataset_clean_ImageNet1k_200"
DEFAULT_DATASET_TYPE="imagenet"

# linear_lbfgs override (optional)
LOGREG_MAX_ITER=1000
LOGREG_C=1.0

# HF datasets
DEFAULT_CUB_SOURCE="hf"
DEFAULT_CUB_HF_DATASET="Donghyun99/CUB-200-2011"
DEFAULT_CUB_HF_TEST_SPLIT="test"

DEFAULT_FOOD_HF_DATASET="ethz/food101"
DEFAULT_FOOD_HF_TEST_SPLIT="validation"

DEFAULT_IMAGENET_HF_DATASET="imagenet-1k"
DEFAULT_IMAGENET_HF_TEST_SPLIT="validation"

# ========================================================
# Internal Logic
# ========================================================

DATASET_DIR="$DEFAULT_DATASET_DIR"
DATASET_TYPE="$DEFAULT_DATASET_TYPE"

CUB_SOURCE="$DEFAULT_CUB_SOURCE"
CUB_HF_DATASET="$DEFAULT_CUB_HF_DATASET"
CUB_HF_TEST_SPLIT="$DEFAULT_CUB_HF_TEST_SPLIT"

FOOD_HF_DATASET="$DEFAULT_FOOD_HF_DATASET"
FOOD_HF_TEST_SPLIT="$DEFAULT_FOOD_HF_TEST_SPLIT"

IMAGENET_HF_DATASET="$DEFAULT_IMAGENET_HF_DATASET"
IMAGENET_HF_TEST_SPLIT="$DEFAULT_IMAGENET_HF_TEST_SPLIT"

CONFIG_FILE=""

# Flags
AUGMENT_FLAG=""
if [ "$ENABLE_AUGMENTATION" = "true" ]; then
  AUGMENT_FLAG="--augment"
fi

TSNE_FLAG="--no_tsne"
if [ "$ENABLE_TSNE" = "true" ]; then
  TSNE_FLAG=""
fi

# Args parsing
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -a|--augment)
      AUGMENT_FLAG="--augment"
      ;;
    --no_tsne)
      TSNE_FLAG="--no_tsne"
      ;;
    --mode)
      # User provided mode will be ignored in the sequential run loop below, 
      # or you can modify the logic to respect it if needed.
      EVAL_MODE="$2"; shift
      ;;
    -t|--type)
      DATASET_TYPE="$2"; shift
      ;;
    --evaluators)
      SELECTED_EVALUATORS="$2"; shift
      ;;
    --config)
      CONFIG_FILE="$2"; shift
      ;;
    --max_iter)
      LOGREG_MAX_ITER="$2"; shift
      ;;
    --C)
      LOGREG_C="$2"; shift
      ;;
    --cub_source)
      CUB_SOURCE="$2"; shift
      ;;
    --cub_hf_dataset)
      CUB_HF_DATASET="$2"; shift
      ;;
    --cub_hf_test_split)
      CUB_HF_TEST_SPLIT="$2"; shift
      ;;
    --food_hf_dataset)
      FOOD_HF_DATASET="$2"; shift
      ;;
    --food_hf_test_split)
      FOOD_HF_TEST_SPLIT="$2"; shift
      ;;
    --imagenet_hf_dataset)
      IMAGENET_HF_DATASET="$2"; shift
      ;;
    --imagenet_hf_test_split)
      IMAGENET_HF_TEST_SPLIT="$2"; shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      DATASET_DIR="$1"
      ;;
  esac
  shift
done

# Check script and data
if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "Error: Python script not found: $PYTHON_SCRIPT" >&2
  exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
  echo "Error: Synthetic dataset directory not found: $DATASET_DIR" >&2
  exit 1
fi

# Build evaluator args safely
EVAL_ARGS=()
if [ -n "${SELECTED_EVALUATORS:-}" ]; then
  read -r -a EVAL_LIST <<< "$SELECTED_EVALUATORS"
  EVAL_ARGS+=(--evaluators "${EVAL_LIST[@]}")
fi

# Define execution timestamp for grouping results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# **********========================================================
# Sequential Execution Loop: linear_lbfgs -> linear_torch
# **********========================================================
#for CURRENT_MODE in "linear_lbfgs" "linear_torch" "partial_ft" "full_ft"; do # Mode 切り替え
for CURRENT_MODE in "linear_torch"; do
#for CURRENT_MODE in "partial_ft" "full_ft"; do # Mode 切り替え
  # Update EVAL_MODE and OUTPUT_DIR for current loop iteration
  EVAL_MODE="$CURRENT_MODE"
  OUTPUT_DIR="$PROJECT_ROOT/evaluation_results_critique/${TIMESTAMP}_${EVAL_MODE}"
  mkdir -p "$OUTPUT_DIR"

  # Generate config json for each run (probe_config.json)
  # This ensures both modes get the config (used primarily by lbfgs or custom logic)
  CONFIG_FILE="$OUTPUT_DIR/probe_config.json"
  cat > "$CONFIG_FILE" <<EOF
{
  "max_iter": ${LOGREG_MAX_ITER},
  "C": ${LOGREG_C}
}
EOF

  echo "========================================================"
  echo " Critical Evaluation Pipeline"
  echo "========================================================"
  echo " Project Root         : $PROJECT_ROOT"
  echo " Python Script        : $PYTHON_SCRIPT"
  echo " Syn Dataset Path     : $DATASET_DIR"
  echo " Output Path          : $OUTPUT_DIR"
  echo " Mode                 : $EVAL_MODE"
  echo " Data Augmentation    : ${AUGMENT_FLAG:-OFF}"
  echo " Selected Evaluators  : ${SELECTED_EVALUATORS:-All(Default)}"
  echo " Config File          : $CONFIG_FILE"
  echo " Dataset Type         : $DATASET_TYPE"
  echo "========================================================"

  # Dependencies check (if lora is used later)
  if [ "$EVAL_MODE" = "lora" ]; then
    pip install -q peft || echo "Warning: PEFT install failed or already satisfied."
  fi

  echo ">>> Running Python Evaluation [$EVAL_MODE]..."

  if [ "$DATASET_TYPE" = "cub" ]; then
    python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --mode "$EVAL_MODE" \
      $AUGMENT_FLAG \
      $TSNE_FLAG \
      "${EVAL_ARGS[@]}" \
      --dataset_type "$DATASET_TYPE" \
      --cub_source "$CUB_SOURCE" \
      --cub_hf_dataset "$CUB_HF_DATASET" \
      --cub_hf_test_split "$CUB_HF_TEST_SPLIT" \
      --config "$CONFIG_FILE"

  elif [ "$DATASET_TYPE" = "food101" ]; then
    python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --mode "$EVAL_MODE" \
      $AUGMENT_FLAG \
      $TSNE_FLAG \
      "${EVAL_ARGS[@]}" \
      --dataset_type "$DATASET_TYPE" \
      --food_hf_dataset "$FOOD_HF_DATASET" \
      --food_hf_test_split "$FOOD_HF_TEST_SPLIT" \
      --config "$CONFIG_FILE"

  else
    python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --mode "$EVAL_MODE" \
      $AUGMENT_FLAG \
      $TSNE_FLAG \
      "${EVAL_ARGS[@]}" \
      --dataset_type "$DATASET_TYPE" \
      --imagenet_hf_dataset "$IMAGENET_HF_DATASET" \
      --imagenet_hf_test_split "$IMAGENET_HF_TEST_SPLIT" \
      --config "$CONFIG_FILE"
  fi

  echo ">>> Finished [$EVAL_MODE]."
  echo ""

done

echo "========================================================"
echo " All Done."
echo " Results are saved in folders starting with: $PROJECT_ROOT/evaluation_results_critique/${TIMESTAMP}_..."
echo "========================================================"