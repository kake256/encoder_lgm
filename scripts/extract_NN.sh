#!/bin/bash
# run_extract_nearest.sh
set -euo pipefail

# ========================================================
# 1. Path & Config
# ========================================================
# プロジェクトルートパス
PROJECT_ROOT="../"

# 基準（クエリ）となる合成画像が入っているディレクトリ
# ※ご提示いただいたパスのルート部分を指定します
DATA="imagenet100_FULL"
SYN_DATASET_DIR="$PROJECT_ROOT/makeData/dataset_clean_$DATA"

# 抽出した実画像の保存先ベースディレクトリ
OUTPUT_BASE_DIR="$PROJECT_ROOT/makeData/${DATA}_prototype_mean"

# 実行するPythonスクリプト
PYTHON_SCRIPT="$PROJECT_ROOT/src/distillation/extract_nearest_real.py"

# 実画像を検索するHuggingFaceデータセット
HF_DATASET="clane9/imagenet-100"
HF_SPLIT="train" # 検索対象は通常trainスプリット

# 抽出に使うモデル（スペース区切りで複数指定可能）
#MODELS="ResNet50 MAE OpenCLIP_RN50 CLIP DINOv2 SigLIP"
MODELS="DINOv1"

# クラス・ジェネレータごとに何枚の実画像を抽出するか
TOP_K=5

# GPU ID
GPU_ID=0

# ========================================================
# 2. Execution
# ========================================================
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

if [ ! -d "$SYN_DATASET_DIR" ]; then
    echo "Error: Synthetic dataset directory not found at $SYN_DATASET_DIR"
    exit 1
fi

echo "========================================================"
echo " Starting Nearest Prototype Extraction"
echo "========================================================"
echo " Script         : $PYTHON_SCRIPT"
echo " Synthetic Data : $SYN_DATASET_DIR"
echo " Output Base    : $OUTPUT_BASE_DIR"
echo " Target Models  : $MODELS"
echo " Top-K Images   : $TOP_K"
echo " HF Dataset     : $HF_DATASET [$HF_SPLIT]"
echo "========================================================"

# コマンドの実行
set -x
CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$PYTHON_SCRIPT" \
    "$SYN_DATASET_DIR" \
    --output_base_dir "$OUTPUT_BASE_DIR" \
    --models $MODELS \
    --top_k $TOP_K \
    --hf_dataset "$HF_DATASET" \
    --hf_split "$HF_SPLIT"
set +x

echo ">>> Extraction Complete."