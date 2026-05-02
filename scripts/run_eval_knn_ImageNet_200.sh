#!/bin/bash
# run_eval_knn.sh
set -euo pipefail

# ========================================================
# User Configuration
# ========================================================

# Evaluators to run
#SELECTED_EVALUATORS="ResNet50 DINOv1 DINOv2 CLIP SigLIP MAE SwAV OpenCLIP_RN50 OpenCLIP_ViT_B32"
SELECTED_EVALUATORS="ResNet50 MAE OpenCLIP_RN50" # Debug用

# KNN Parameter (Default: 10)
K_NEIGHBORS=10

# Baseline & Centroid Parameters
# 複数のベースラインをカンマ区切りで指定可能 (例: "5", "5,200", "5,200,1300")
BASELINE_SHOTS="5,200,1500"
# ロードする実画像の最大数 (BASELINE_SHOTSの最大値以上にする必要があります)
MAX_TRAIN_SAMPLES=1500

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Python Script Path
PYTHON_SCRIPT="$PROJECT_ROOT/src/evaluate_knn.py"

# Default Paths (Switch these or override via command line)
# Example: CUB
DEFAULT_DATASET_DIR="$PROJECT_ROOT/makeData/dataset_clean_ImageNet1k_200"
DEFAULT_DATASET_TYPE="imagenet"

# Example: Food-101
#DEFAULT_DATASET_DIR="$PROJECT_ROOT/makeData/dataset_clean_food101"
#DEFAULT_DATASET_TYPE="food101"

# HF Settings
DEFAULT_CUB_HF="Donghyun99/CUB-200-2011"
DEFAULT_CUB_SPLIT="test"

DEFAULT_FOOD_HF="ethz/food101"
DEFAULT_FOOD_SPLIT="validation"

DEFAULT_IMAGENET_HF="imagenet-1k"
DEFAULT_IMAGENET_SPLIT="validation"

# ========================================================
# Internal Logic
# ========================================================

DATASET_DIR="$DEFAULT_DATASET_DIR"
DATASET_TYPE="$DEFAULT_DATASET_TYPE"

CUB_HF="$DEFAULT_CUB_HF"
CUB_SPLIT="$DEFAULT_CUB_SPLIT"

FOOD_HF="$DEFAULT_FOOD_HF"
FOOD_SPLIT="$DEFAULT_FOOD_SPLIT"

IMAGENET_HF="$DEFAULT_IMAGENET_HF"
IMAGENET_SPLIT="$DEFAULT_IMAGENET_SPLIT"

# UMAPフラグ (デフォルトでOFFにする)
UMAP_FLAG=""

# Args parsing
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -t|--type)
      DATASET_TYPE="$2"; shift
      ;;
    --evaluators)
      SELECTED_EVALUATORS="$2"; shift
      ;;
    --k)
      K_NEIGHBORS="$2"; shift
      ;;
    --shots)          # 複数指定時は "5,200,1300" のようにダブルクォーテーションで囲む
      BASELINE_SHOTS="$2"; shift
      ;;
    --max_train)
      MAX_TRAIN_SAMPLES="$2"; shift
      ;;
    --umap)           # 引数で --umap を渡した時だけ有効になる
      UMAP_FLAG="--umap"
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

# Check script
if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "Error: Python script not found: $PYTHON_SCRIPT" >&2
  exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
  echo "Error: Dataset directory not found: $DATASET_DIR" >&2
  exit 1
fi

# Build evaluator args
EVAL_ARGS=()
if [ -n "${SELECTED_EVALUATORS:-}" ]; then
  read -r -a EVAL_LIST <<< "$SELECTED_EVALUATORS"
  EVAL_ARGS+=(--evaluators "${EVAL_LIST[@]}")
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_ROOT/evaluation_results_knn/${DATASET_TYPE}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo " KNN Evaluation Pipeline"
echo "========================================================"
echo " Script         : $PYTHON_SCRIPT"
echo " Dataset Path   : $DATASET_DIR"
echo " Type           : $DATASET_TYPE"
echo " K Neighbors    : $K_NEIGHBORS"
echo " Baseline Shots : $BASELINE_SHOTS"
echo " Max Train Load : $MAX_TRAIN_SAMPLES"
echo " Evaluators     : ${SELECTED_EVALUATORS}"
echo " Output Dir     : $OUTPUT_DIR"
echo " UMAP Enabled   : ${UMAP_FLAG:-(None)}"
echo "========================================================"

echo ">>> Running KNN Evaluation..."

if [ "$DATASET_TYPE" = "cub" ]; then
  python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_type "$DATASET_TYPE" \
    --hf_dataset "$CUB_HF" \
    --hf_split "$CUB_SPLIT" \
    --k_neighbors "$K_NEIGHBORS" \
    --baseline_shots "$BASELINE_SHOTS" \
    --max_train_samples "$MAX_TRAIN_SAMPLES" \
    --save_analysis \
    $UMAP_FLAG \
    "${EVAL_ARGS[@]}"

elif [ "$DATASET_TYPE" = "food101" ]; then
  python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_type "$DATASET_TYPE" \
    --hf_dataset "$FOOD_HF" \
    --hf_split "$FOOD_SPLIT" \
    --k_neighbors "$K_NEIGHBORS" \
    --baseline_shots "$BASELINE_SHOTS" \
    --max_train_samples "$MAX_TRAIN_SAMPLES" \
    --save_analysis \
    $UMAP_FLAG \
    "${EVAL_ARGS[@]}"

else
  python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_type "$DATASET_TYPE" \
    --hf_dataset "$IMAGENET_HF" \
    --hf_split "$IMAGENET_SPLIT" \
    --k_neighbors "$K_NEIGHBORS" \
    --baseline_shots "$BASELINE_SHOTS" \
    --max_train_samples "$MAX_TRAIN_SAMPLES" \
    --save_analysis \
    $UMAP_FLAG \
    "${EVAL_ARGS[@]}"
fi

echo ">>> Done."
echo "Results saved to: $OUTPUT_DIR"