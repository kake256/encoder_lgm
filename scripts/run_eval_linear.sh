#!/bin/bash
set -e

# ========================================================
# ユーザー設定: データ拡張スイッチ
# ========================================================
# ここを "true" にするとデフォルトでデータ拡張がONになります
# "false" にするとOFFになります（引数 -a で強制ON可能）
ENABLE_AUGMENTATION="true"

# ========================================================
# 設定とパス解決
# ========================================================

# 1. このスクリプトがあるディレクトリ (scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 2. プロジェクトのルートディレクトリ (scripts/ の一つ上)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 3. Pythonスクリプトのパス
PYTHON_SCRIPT="$PROJECT_ROOT/src/evaluate_linear.py"

# 4. データセットのデフォルトパス
DEFAULT_DATASET_DIR="$PROJECT_ROOT/makeData/dataset_clean"

# ========================================================
# 引数解析と設定の反映
# ========================================================

# 内部設定に基づいて初期値を決定
if [ "$ENABLE_AUGMENTATION" = "true" ]; then
    AUGMENT_FLAG="--augment"
    AUGMENT_STATUS="ON (Config)"
else
    AUGMENT_FLAG=""
    AUGMENT_STATUS="OFF"
fi

# 初期データセットパス
DATASET_DIR="$DEFAULT_DATASET_DIR"

# 引数をループで処理 (内部設定より引数を優先または追加)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -a|--augment)
            AUGMENT_FLAG="--augment"
            AUGMENT_STATUS="ON (Argument)"
            ;;
        -*) # 未知のオプション
            echo "Unknown option: $1"
            echo "Usage: bash run_eval.sh [dataset_path] [--augment|-a]"
            exit 1
            ;;
        *) # 位置引数はデータセットパスとみなす
            DATASET_DIR="$1"
            ;;
    esac
    shift
done

# ========================================================
# 出力ディレクトリ設定 (タイムスタンプ付き)
# ========================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_ROOT/evaluation_results_critique/${TIMESTAMP}"

# ========================================================
# 実行前チェック
# ========================================================

echo "========================================================"
echo " Critical Evaluation Pipeline"
echo "========================================================"
echo " Project Root      : $PROJECT_ROOT"
echo " Python Script     : $PYTHON_SCRIPT"
echo " Dataset Path      : $DATASET_DIR"
echo " Output Path       : $OUTPUT_DIR"
echo " Data Augmentation : $AUGMENT_STATUS"
echo "========================================================"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory not found at $DATASET_DIR"
    exit 1
fi

# ========================================================
# 実行
# ========================================================

echo ">>> Installing Dependencies (if needed)..."
pip install transformers datasets scikit-learn pandas matplotlib sentencepiece tqdm pillow torchvision --quiet

echo ">>> Running Python Evaluation..."
# 引数を渡して実行
python3 "$PYTHON_SCRIPT" "$DATASET_DIR" --output_dir "$OUTPUT_DIR" $AUGMENT_FLAG

echo "========================================================"
echo " All Done."
echo " Results are saved in: $OUTPUT_DIR"
echo "========================================================"