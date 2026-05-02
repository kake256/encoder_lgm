#!/bin/bash
# run_eval_all_combined.sh
set -euo pipefail

# ========================================================
# 1. Common Configuration (共通設定)
# ========================================================

# 評価に使用するモデル
# SELECTED_EVALUATORS="ResNet50 MAE SwAV OpenCLIP_RN50 OpenCLIP_ViT_B32"
SELECTED_EVALUATORS="ResNet50 MAE OpenCLIP_RN50"

# フラグ設定
ENABLE_AUGMENTATION="true"
ENABLE_TSNE="false"

# --- [追加] Data Augmentation Strategy ---
# GB10のような高速GPUでは "precompute" が推奨です (学習中のCPUボトルネックを回避)
# Options: "none", "precompute", "on_the_fly"
#AUG_STRATEGY="precompute"
AUG_STRATEGY="on_the_fly"

# precompute時の倍率 (例: 5枚の画像を20倍に拡張して100枚分の特徴量としてキャッシュする)
AUG_EXPANSION=20

# --- [追加] 平均化のための試行回数 ---
# 複数回実行し、平均と分散(Std)を記録します
NUM_TRIALS=5

# --- 実画像の比較枚数設定 ---
# 1. FewShot(5), 2. 指定値(200), 3. 上限値(1300)
REAL_IMAGE_COUNTS="5" #"200 1300"
REAL_TEST_COUNT=50

# --- 合成画像の段階的検証設定 ---
SYN_COUNTS="1 3 5" #"1 3 5"

# --- Mixソースの定義 ---
# ここではスペース区切りのリストとして定義します
#MIX_SET_SINGLE="Only_V1 Only_V2 Only_CLIP Only_SigLIP"
#MIX_SET_MULTI="Hybrid_V1_V2 Hybrid_CLIP_SigLIP Hybrid_V1_CLIP Hybrid_V1_SigLIP Hybrid_V2_CLIP Hybrid_V2_SigLIP"
#MIX_SET_ALL="$MIX_SET_SINGLE $MIX_SET_MULTI"
MIX_SET_SINGLE=""
MIX_SET_MULTI=""
MIX_SET_ALL=""

# Pythonスクリプトのパス
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$PROJECT_ROOT/src/evaluate_linear2.py"

# Linear Probe Config
LOGREG_MAX_ITER=1000
LOGREG_C=1.0

# 前提チェック
if [ ! -f "$PYTHON_SCRIPT" ]; then echo "Error: Python script not found at $PYTHON_SCRIPT"; exit 1; fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ========================================================
# 2. Execution Function (実行用関数)
# ========================================================

run_dataset_cycle() {
    local DS_NAME="$1"       # ディレクトリ名のサフィックス (例: ImageNet1k_200)
    local DS_TYPE="$2"       # 引数 --dataset_type (例: imagenet)
    local HF_DS="$3"         # HuggingFace Dataset Name
    local HF_SPLIT="$4"      # HuggingFace Split

    # データセットディレクトリのパス構築
    local DATASET_DIR="$PROJECT_ROOT/makeData/dataset_clean_${DS_NAME}"

    if [ ! -d "$DATASET_DIR" ]; then
        echo "--------------------------------------------------------"
        echo " [SKIP] Directory not found: $DATASET_DIR"
        echo "--------------------------------------------------------"
        return
    fi

    echo "********************************************************"
    echo " STARTING DATASET: $DS_NAME ($DS_TYPE)"
    echo "********************************************************"

    # データセットごとの引数分岐
    local EXTRA_ARGS=""
    if [ "$DS_TYPE" == "imagenet" ]; then
        EXTRA_ARGS="--imagenet_hf_dataset $HF_DS --imagenet_hf_test_split $HF_SPLIT"
    elif [ "$DS_TYPE" == "food101" ]; then
        EXTRA_ARGS="--food_hf_dataset $HF_DS --food_hf_test_split $HF_SPLIT"
    elif [ "$DS_TYPE" == "cub" ]; then
        EXTRA_ARGS="--cub_hf_dataset $HF_DS --cub_hf_test_split $HF_SPLIT"
    fi

    # --------------------------------------------------------
    # Mix設定をJSON文字列に変換する処理
    # --------------------------------------------------------
    # スペース区切りのリストを、JSONの配列形式 ["A", "B", ...] に変換します
    local LIST_SINGLE=$(echo $MIX_SET_SINGLE | sed 's/ /", "/g')
    local LIST_MULTI=$(echo $MIX_SET_MULTI | sed 's/ /", "/g')
    local LIST_ALL=$(echo $MIX_SET_ALL | sed 's/ /", "/g')

    # Pythonに渡すJSON文字列を構築
    local MIX_JSON_STR="{\"Mix_Single\": [\"$LIST_SINGLE\"], \"Mix_Multi\": [\"$LIST_MULTI\"], \"Mix_All\": [\"$LIST_ALL\"]}"

    # モード固定
    local EVAL_MODE="linear_torch"

    # 出力先ディレクトリ (戦略も含めてフォルダ分けしておくと便利です)
    local OUTPUT_DIR="$PROJECT_ROOT/evaluation_results_critique/${DS_NAME}/${TIMESTAMP}_${EVAL_MODE}_Combined_${AUG_STRATEGY}"
    mkdir -p "$OUTPUT_DIR"

    # Config生成
    local CONFIG_FILE="$OUTPUT_DIR/probe_config.json"
    echo "{\"max_iter\": $LOGREG_MAX_ITER, \"C\": $LOGREG_C}" > "$CONFIG_FILE"

    echo "--------------------------------------------------------"
    echo " Run Config: $DS_NAME | Trials: $NUM_TRIALS"
    echo " Strategy: $AUG_STRATEGY (Expansion: ${AUG_EXPANSION}x)"
    echo " Mix Strategies: Single, Multi, All (Running simultaneously)"
    echo "--------------------------------------------------------"
    echo " Output: $OUTPUT_DIR"
    
    # Python実行
    # --augment フラグに加え、--aug_strategy と --aug_expansion を渡します
    python3 "$PYTHON_SCRIPT" "$DATASET_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --mode "$EVAL_MODE" \
        --real_counts $REAL_IMAGE_COUNTS \
        --max_real_test $REAL_TEST_COUNT \
        --syn_counts $SYN_COUNTS \
        --mix_json "$MIX_JSON_STR" \
        --num_trials $NUM_TRIALS \
        --augment \
        --aug_strategy "$AUG_STRATEGY" \
        --aug_expansion $AUG_EXPANSION \
        --no_tsne \
        --evaluators $SELECTED_EVALUATORS \
        --dataset_type "$DS_TYPE" \
        $EXTRA_ARGS \
        --config "$CONFIG_FILE"

    echo "Finished cycle for $DS_NAME"
    echo ""
}

# ========================================================
# 3. Main Execution Sequence (実行順序)
# ========================================================

# 1. ImageNet
# ディレクトリ: makeData/dataset_clean_ImageNet1k_200
run_dataset_cycle "ImageNet1k_200" "imagenet" "imagenet-1k" "validation"

# 2. Food101
# ディレクトリ: makeData/dataset_clean_food101
run_dataset_cycle "food101" "food101" "ethz/food101" "validation"

# 3. CUB
# ディレクトリ: makeData/dataset_clean_CUB
run_dataset_cycle "CUB" "cub" "Donghyun99/CUB-200-2011" "test"

echo "========================================================"
echo " All Evaluation Cycles Completed."
echo "========================================================"