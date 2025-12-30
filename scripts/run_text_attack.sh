#!/bin/bash
set -e

# ========================================================
# 1. 環境設定
# ========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
RESULTS_BASE_DIR="$PROJECT_DIR/classification_results"
EXPERIMENT_ROOT_DIR="$RESULTS_BASE_DIR/text_attack_experiments" # 出力先を少し変更

# ★実行するPythonスクリプト名 (統合版のファイル名に合わせてください)
PYTHON_SCRIPT_NAME="main_text_overlay.py"

mkdir -p "$EXPERIMENT_ROOT_DIR"
export PYTHONPATH="$SRC_DIR:$PROJECT_DIR:$PYTHONPATH"

# --------------------------------------------------------
# ★【追加】EasyOCRモデルの事前ダウンロード (競合回避用)
# 並列実行時に同時にダウンロードが走ってエラーになるのを防ぐため、ここで一度実行しておきます
# --------------------------------------------------------
echo ">>> Checking/Downloading EasyOCR models to avoid race conditions..."
python3 -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"
echo ">>> EasyOCR models are ready."
# --------------------------------------------------------

# ========================================================
# 2. データセット設定
# ========================================================

# --- ImageNet + Text Overlay 設定 ---
DATASET_TYPE="imagenet"
# ImageNetの画像をロードして文字を乗せるため、DATA_ROOTは不要(HuggingFace経由)ですが
# コードの互換性のために残します
DATA_ROOT="" 

# ターゲットクラス (リスト形式で記述)
# ImageNetのクラスIDを指定 (例: 9=Ostrich, 950=Orange, 1=Goldfish, 108=Jellyfish, 404=Airliner)
TARGET_CLASSES="9 950 1 108 404"

# ========================================================
# 3. 実験パラメータ
# ========================================================

# --- ★新規: 文字攻撃(Text Overlay)パラメータ ---
OVERLAY_TEXT="dog"      # 画像に書き込む文字 (Conflictを起こす文字)
TEXT_COLOR="red"         # 文字の色
FONT_SCALE=0.15          # 画像サイズに対するフォントサイズの割合

# モデルリスト
ENCODER_NAMES=(
    "facebook/dino-vitb16"
    "facebook/dinov2-base"
    "openai/clip-vit-base-patch16"
)

# 射影層の次元バリエーション
PROJ_DIM_LIST=(
    "0 0 0"             # 射影なし
    "2048 2048 2048"    # 高次元
)

# 実験設定リスト
EXPERIMENTS=(
    "Only_v1:1.0,0.0,0.0"
    "Only_v2:0.0,1.0,0.0"
    "Only_CLIP:0.0,0.0,1.0"
    "AND_v1_CLIP:1.0,0.0,1.0"
    "AND_v2_CLIP:0.0,1.0,1.0"
    "AND_v1_v2:1.0,1.0,0.0"
    "AND_All_Three:1.0,1.0,1.0"
)

# 固定パラメータ
ITERATIONS=4000
LR=0.002
IMAGE_SIZE=224
NUM_REF_IMAGES=10
AUGS_PER_STEP=32
WEIGHT_TV=0.00025

# 並列設定 (GPU枚数に合わせて調整してください)
MAX_JOBS=2

# ========================================================
# 4. 実行ループ
# ========================================================
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="$EXPERIMENT_ROOT_DIR/$TIMESTAMP"
mkdir -p "$BASE_OUTPUT_DIR"

echo "=========================================="
echo "Starting Parallel Text Attack Experiments"
echo "Target Text: $OVERLAY_TEXT"
echo "Output: $BASE_OUTPUT_DIR"
echo "Max Jobs: $MAX_JOBS"
echo "=========================================="

# --- 次元設定のループ ---
for PROJ_DIMS in "${PROJ_DIM_LIST[@]}"; do
    DIM_DIR_NAME="dim_${PROJ_DIMS// /_}"
    BATCH_DIR="$BASE_OUTPUT_DIR/$DIM_DIR_NAME"
    mkdir -p "$BATCH_DIR"

    echo ""
    echo "------------------------------------------"
    echo ">>> Running with Projection Dims: [ $PROJ_DIMS ]"
    echo "------------------------------------------"

    # ターゲットクラスのリスト化
    if [ -z "$TARGET_CLASSES" ]; then
        CLASS_LOOP=("ALL")
        echo "   Target Classes is empty. Running all classes (Not Recommended for this mode)."
    else
        CLASS_LOOP=($TARGET_CLASSES)
    fi

    PIDS=()
    
    # --- クラスごとのループ (並列処理) ---
    for CLASS_ID in "${CLASS_LOOP[@]}"; do
        
        # GPU IDの割り当て (0 または 1)
        GPU_ID=$((${#PIDS[@]} % MAX_JOBS))
        
        # 引数とログファイルの設定
        if [ "$CLASS_ID" == "ALL" ]; then
            TARGET_ARG=""
            LOG_FILENAME="run_all_classes.log"
        else
            TARGET_ARG="--target_classes $CLASS_ID"
            LOG_FILENAME="run_class_${CLASS_ID}.log"
        fi
        
        LOG_FILE_PATH="$BATCH_DIR/$LOG_FILENAME"

        # 出力制御のロジック (GPU 0 は表示、GPU 1 は非表示)
        if [ "$GPU_ID" -eq 0 ]; then
            REDIRECT=""
            echo "   [GPU $GPU_ID] Class: $CLASS_ID (Log visible)"
        else
            REDIRECT="> /dev/null 2>&1"
            echo "   [GPU $GPU_ID] Class: $CLASS_ID (Log hidden -> $LOG_FILENAME)"
        fi

        # Python実行
        # 新しい引数 (--overlay_text 等) を追加しています
        CMD="cd \"$SRC_DIR\" && CUDA_VISIBLE_DEVICES=$GPU_ID python3 $PYTHON_SCRIPT_NAME \
            --encoder_names ${ENCODER_NAMES[@]} \
            --projection_dims $PROJ_DIMS \
            --experiments ${EXPERIMENTS[@]} \
            $TARGET_ARG \
            --dataset_type \"$DATASET_TYPE\" \
            --data_root \"$DATA_ROOT\" \
            --output_dir \"$BATCH_DIR\" \
            --num_iterations $ITERATIONS \
            --lr $LR \
            --image_size $IMAGE_SIZE \
            --num_ref_images $NUM_REF_IMAGES \
            --augs_per_step $AUGS_PER_STEP \
            --weight_tv $WEIGHT_TV \
            --overlay_text \"$OVERLAY_TEXT\" \
            --text_color \"$TEXT_COLOR\" \
            --font_scale $FONT_SCALE \
            --log_file \"$LOG_FILE_PATH\" $REDIRECT &"

        eval "$CMD"
        PIDS+=($!)

        # 指定した並列数(MAX_JOBS)に達したら完了を待つ
        if [ ${#PIDS[@]} -ge $MAX_JOBS ]; then
            echo "   >>> Waiting for batch completion..."
            for pid in "${PIDS[@]}"; do
                wait "$pid"
            done
            PIDS=() # PIDリストをリセット
            echo "   >>> Batch finished. Proceeding."
        fi

    done

    # ループ終了時に残っているジョブがあれば待つ
    if [ ${#PIDS[@]} -gt 0 ]; then
        echo "   >>> Waiting for remaining jobs..."
        for pid in "${PIDS[@]}"; do
            wait "$pid"
        done
        PIDS=()
    fi

    echo ">>> Completed Projection Dims: $PROJ_DIMS"
done

echo "=========================================="
echo "All Experiments Completed."
echo "Results located at: $BASE_OUTPUT_DIR"
echo "=========================================="