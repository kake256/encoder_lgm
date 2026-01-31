#!/bin/bash
# run_eval_sequential_loop.sh
set -euo pipefail

# ========================================================
# User Configuration
# ========================================================

# Evaluators (space-separated names, must match keys in AVAILABLE_EVAL_MODELS)
# 【変更点1】評価対象モデルを指定（ResNet50, OpenCLIP_RN50, MAE + 新規軽量モデル群）
SELECTED_EVALUATORS="ResNet50 OpenCLIP_RN50 MAE MobileNetV2_050 MobileNetV3_S GhostNet_100 ConvNeXt_Atto EfficientNet_B0 ResNet10t ResNet18"

#ENABLE_AUGMENTATION="false"
ENABLE_AUGMENTATION="true"
ENABLE_TSNE="false"  # kept for compatibility, not used

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Update this to point to your latest python script
PYTHON_SCRIPT="$PROJECT_ROOT/src/evaluate_linear2.py"

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

# Loop Configurations
# (dataset_dir, output_subdir) pair
DATASETS="$PROJECT_ROOT/makeData/dataset_clean_ImageNet1k_200:ImageNet1k_200"

# ========================================================
# Internal Logic
# ========================================================

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

# Args parsing (basic override)
DATASET_TYPE="imagenet"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -a|--augment)
      AUGMENT_FLAG="--augment"
      ;;
    --no_tsne)
      TSNE_FLAG="--no_tsne"
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
      DEFAULT_CUB_SOURCE="$2"; shift
      ;;
    --cub_hf_dataset)
      DEFAULT_CUB_HF_DATASET="$2"; shift
      ;;
    --cub_hf_test_split)
      DEFAULT_CUB_HF_TEST_SPLIT="$2"; shift
      ;;
    --food_hf_dataset)
      DEFAULT_FOOD_HF_DATASET="$2"; shift
      ;;
    --food_hf_test_split)
      DEFAULT_FOOD_HF_TEST_SPLIT="$2"; shift
      ;;
    --imagenet_hf_dataset)
      DEFAULT_IMAGENET_HF_DATASET="$2"; shift
      ;;
    --imagenet_hf_test_split)
      DEFAULT_IMAGENET_HF_TEST_SPLIT="$2"; shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      # Ignore positional args for simplicity in loop
      ;;
  esac
  shift
done

# Check script
if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "Error: Python script not found: $PYTHON_SCRIPT" >&2
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
# Main Loop: Iterate over Datasets -> Modes
# **********========================================================

for DATASET_INFO in "${DATASETS[@]}"; do
    IFS=":" read -r DATASET_DIR SUBDIR_NAME <<< "$DATASET_INFO"

    if [ ! -d "$DATASET_DIR" ]; then
        echo "Warning: Dataset directory not found: $DATASET_DIR. Skipping..."
        continue
    fi

    # Inner Loop: linear_torch -> scratch
    #for CURRENT_MODE in "linear_lbfgs" "linear_torch" "partial_ft" "full_ft"; do # Mode 切り替え
    for CURRENT_MODE in "full_ft"; do 
        EVAL_MODE="$CURRENT_MODE"
        
        # Output directory includes SUBDIR_NAME (e.g. ImageNet1k_200) to separate results
        OUTPUT_DIR="$PROJECT_ROOT/evaluation_results_critique/${SUBDIR_NAME}/${TIMESTAMP}_${EVAL_MODE}"
        mkdir -p "$OUTPUT_DIR"

        # Generate config json
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
        echo " Dataset Subdir       : $SUBDIR_NAME"
        echo " Data Augmentation    : ${AUGMENT_FLAG:-OFF}"
        echo " Selected Evaluators  : ${SELECTED_EVALUATORS:-All(Default)}"
        echo " Config File          : $CONFIG_FILE"
        echo " Dataset Type         : $DATASET_TYPE"
        echo "========================================================"

        # Dependencies check (if lora is used later)
        if [ "$EVAL_MODE" = "lora" ]; then
            pip install -q peft || echo "Warning: PEFT install failed or already satisfied."
        fi

        echo ">>> Running Python Evaluation [$EVAL_MODE] on [$SUBDIR_NAME]..."

        if [ "$DATASET_TYPE" = "cub" ]; then
            python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --mode "$EVAL_MODE" \
            $AUGMENT_FLAG \
            $TSNE_FLAG \
            "${EVAL_ARGS[@]}" \
            --dataset_type "$DATASET_TYPE" \
            --cub_source "$DEFAULT_CUB_SOURCE" \
            --cub_hf_dataset "$DEFAULT_CUB_HF_DATASET" \
            --cub_hf_test_split "$DEFAULT_CUB_HF_TEST_SPLIT" \
            --config "$CONFIG_FILE"

        elif [ "$DATASET_TYPE" = "food101" ]; then
            python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --mode "$EVAL_MODE" \
            $AUGMENT_FLAG \
            $TSNE_FLAG \
            "${EVAL_ARGS[@]}" \
            --dataset_type "$DATASET_TYPE" \
            --food_hf_dataset "$DEFAULT_FOOD_HF_DATASET" \
            --food_hf_test_split "$DEFAULT_FOOD_HF_TEST_SPLIT" \
            --config "$CONFIG_FILE"

        else
            python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --mode "$EVAL_MODE" \
            $AUGMENT_FLAG \
            $TSNE_FLAG \
            "${EVAL_ARGS[@]}" \
            --dataset_type "$DATASET_TYPE" \
            --imagenet_hf_dataset "$DEFAULT_IMAGENET_HF_DATASET" \
            --imagenet_hf_test_split "$DEFAULT_IMAGENET_HF_TEST_SPLIT" \
            --config "$CONFIG_FILE"
        fi

        echo ">>> Finished [$EVAL_MODE] on [$SUBDIR_NAME]."
        echo ""
    done
done

echo "========================================================"
echo " All Done."
echo " Results are saved in folders starting with: $PROJECT_ROOT/evaluation_results_critique/"
echo "========================================================"




# 2
#!/bin/bash
# run_eval_sequential_loop.sh
set -euo pipefail

# ========================================================
# User Configuration
# ========================================================

# Evaluators (space-separated names, must match keys in AVAILABLE_EVAL_MODELS)
# 【変更点1】評価対象モデルを指定（ResNet50, OpenCLIP_RN50, MAE + 新規軽量モデル群）
SELECTED_EVALUATORS="ResNet50 OpenCLIP_RN50 MAE MobileNetV2_050 MobileNetV3_S GhostNet_100 ConvNeXt_Atto EfficientNet_B0 ResNet10t ResNet18"

#ENABLE_AUGMENTATION="false"
ENABLE_AUGMENTATION="true"
ENABLE_TSNE="false"  # kept for compatibility, not used

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Update this to point to your latest python script
PYTHON_SCRIPT="$PROJECT_ROOT/src/evaluate_linear2.py"

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

# Loop Configurations
# (dataset_dir, output_subdir) pair
DATASETS="$PROJECT_ROOT/makeData/dataset_clean:ImageNet1k"

# ========================================================
# Internal Logic
# ========================================================

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

# Args parsing (basic override)
DATASET_TYPE="imagenet"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -a|--augment)
      AUGMENT_FLAG="--augment"
      ;;
    --no_tsne)
      TSNE_FLAG="--no_tsne"
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
      DEFAULT_CUB_SOURCE="$2"; shift
      ;;
    --cub_hf_dataset)
      DEFAULT_CUB_HF_DATASET="$2"; shift
      ;;
    --cub_hf_test_split)
      DEFAULT_CUB_HF_TEST_SPLIT="$2"; shift
      ;;
    --food_hf_dataset)
      DEFAULT_FOOD_HF_DATASET="$2"; shift
      ;;
    --food_hf_test_split)
      DEFAULT_FOOD_HF_TEST_SPLIT="$2"; shift
      ;;
    --imagenet_hf_dataset)
      DEFAULT_IMAGENET_HF_DATASET="$2"; shift
      ;;
    --imagenet_hf_test_split)
      DEFAULT_IMAGENET_HF_TEST_SPLIT="$2"; shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      # Ignore positional args for simplicity in loop
      ;;
  esac
  shift
done

# Check script
if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "Error: Python script not found: $PYTHON_SCRIPT" >&2
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
# Main Loop: Iterate over Datasets -> Modes
# **********========================================================

for DATASET_INFO in "${DATASETS[@]}"; do
    IFS=":" read -r DATASET_DIR SUBDIR_NAME <<< "$DATASET_INFO"

    if [ ! -d "$DATASET_DIR" ]; then
        echo "Warning: Dataset directory not found: $DATASET_DIR. Skipping..."
        continue
    fi

    # Inner Loop: linear_torch -> scratch
    #for CURRENT_MODE in "linear_lbfgs" "linear_torch" "partial_ft" "full_ft"; do # Mode 切り替え
    for CURRENT_MODE in "full_ft" "scratch"; do 
        EVAL_MODE="$CURRENT_MODE"
        
        # Output directory includes SUBDIR_NAME (e.g. ImageNet1k) to separate results
        OUTPUT_DIR="$PROJECT_ROOT/evaluation_results_critique/${SUBDIR_NAME}/${TIMESTAMP}_${EVAL_MODE}"
        mkdir -p "$OUTPUT_DIR"

        # Generate config json
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
        echo " Dataset Subdir       : $SUBDIR_NAME"
        echo " Data Augmentation    : ${AUGMENT_FLAG:-OFF}"
        echo " Selected Evaluators  : ${SELECTED_EVALUATORS:-All(Default)}"
        echo " Config File          : $CONFIG_FILE"
        echo " Dataset Type         : $DATASET_TYPE"
        echo "========================================================"

        # Dependencies check (if lora is used later)
        if [ "$EVAL_MODE" = "lora" ]; then
            pip install -q peft || echo "Warning: PEFT install failed or already satisfied."
        fi

        echo ">>> Running Python Evaluation [$EVAL_MODE] on [$SUBDIR_NAME]..."

        if [ "$DATASET_TYPE" = "cub" ]; then
            python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --mode "$EVAL_MODE" \
            $AUGMENT_FLAG \
            $TSNE_FLAG \
            "${EVAL_ARGS[@]}" \
            --dataset_type "$DATASET_TYPE" \
            --cub_source "$DEFAULT_CUB_SOURCE" \
            --cub_hf_dataset "$DEFAULT_CUB_HF_DATASET" \
            --cub_hf_test_split "$DEFAULT_CUB_HF_TEST_SPLIT" \
            --config "$CONFIG_FILE"

        elif [ "$DATASET_TYPE" = "food101" ]; then
            python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --mode "$EVAL_MODE" \
            $AUGMENT_FLAG \
            $TSNE_FLAG \
            "${EVAL_ARGS[@]}" \
            --dataset_type "$DATASET_TYPE" \
            --food_hf_dataset "$DEFAULT_FOOD_HF_DATASET" \
            --food_hf_test_split "$DEFAULT_FOOD_HF_TEST_SPLIT" \
            --config "$CONFIG_FILE"

        else
            python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --mode "$EVAL_MODE" \
            $AUGMENT_FLAG \
            $TSNE_FLAG \
            "${EVAL_ARGS[@]}" \
            --dataset_type "$DATASET_TYPE" \
            --imagenet_hf_dataset "$DEFAULT_IMAGENET_HF_DATASET" \
            --imagenet_hf_test_split "$DEFAULT_IMAGENET_HF_TEST_SPLIT" \
            --config "$CONFIG_FILE"
        fi

        echo ">>> Finished [$EVAL_MODE] on [$SUBDIR_NAME]."
        echo ""
    done
done

echo "========================================================"
echo " All Done."
echo " Results are saved in folders starting with: $PROJECT_ROOT/evaluation_results_critique/"
echo "========================================================"



