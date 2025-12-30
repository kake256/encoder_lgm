#!/bin/bash
# 元の設定に近づけたver
set -e

# ========================================================
# 1. 環境設定
# ========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
RESULTS_BASE_DIR="$PROJECT_DIR/classification_results"

# [変更] 出力先を "high_quality" に変更
#EXPERIMENT_ROOT_DIR="$RESULTS_BASE_DIR/gene_experiment_4models_fast"
EXPERIMENT_ROOT_DIR="$RESULTS_BASE_DIR/gene_experiment_4models_high_quality"

mkdir -p "$EXPERIMENT_ROOT_DIR"
export PYTHONPATH="$SRC_DIR:$PROJECT_DIR:$PYTHONPATH"

# ========================================================
# 2. データセット設定
# ========================================================
DATASET_TYPE="imagenet"
DATA_ROOT="" 

# ターゲットクラス
# 【ターゲットクラス一覧 (計20種)】
# 1   : 金魚 (Goldfish) - 色彩・透明感
# 9   : ダチョウ (Ostrich) - 生物・詳細
# 130 : フラミンゴ (Flamingo) - 色彩・群れ
# 250 : シベリアンハスキー (Siberian husky) - 毛並み・犬
# 340 : シマウマ (Zebra) - 強烈なテクスチャ
# 404 : 旅客機 (Airliner) - 人工物・形状
# 483 : 城 (Castle) - 建築物・複雑な構造
# 598 : 蜂の巣 (Honeycomb) - 幾何学パターン
# 682 : オベリスク (Obelisk) - シンプルな石造建築
# 776 : サックス (Saxophone) - 金属光沢・複雑な形状
# 805 : サッカーボール (Soccer ball) - 幾何学模様
# 815 : クモの巣 (Spider web) - 非常に細い線・透明感
# 850 : テディベア (Teddy bear) - 毛の質感・ぬいぐるみ
# 920 : 信号機 (Traffic light) - 人工物・発光
# 937 : ブロッコリー (Broccoli) - 細かい粒状の質感
# 948 : 青リンゴ (Granny Smith) - シンプル物体・色
# 973 : サンゴ礁 (Coral reef) - 複雑な自然形状・色彩
# 980 : 火山 (Volcano) - 不定形・爆発・煙
# 993 : ヒナギク (Daisy) - 花弁・自然
# 999 : トイレットペーパー (Toilet tissue) - 白一色・質感

TARGET_CLASSES="9 130"
#TARGET_CLASSES="1 9 130 250 340 404 483 598 682 776 805 815 850 920 937 948 973 980 993 999"


# ========================================================
# 3. 実験パラメータ
# ========================================================

ENCODER_NAMES=(
    "facebook/dino-vitb16"
    "facebook/dinov2-base"
    "openai/clip-vit-base-patch16"
    "google/siglip-base-patch16-224"
)

PROJ_DIM_LIST=("0 0 0 0")

EXPERIMENTS=(
    "Only_V1:1.0,0.0,0.0,0.0"
    "Only_V2:0.0,1.0,0.0,0.0"
    "Only_CLIP:0.0,0.0,1.0,0.0"
    "Only_SigLIP:0.0,0.0,0.0,1.0"
    "Hybrid_V1_CLIP:1.0,0.0,1.0,0.0"
    "Hybrid_V1_SigLIP:1.0,0.0,0.0,1.0"
    "Hybrid_V2_CLIP:0.0,1.0,1.0,0.0"
    "Hybrid_V2_SigLIP:0.0,1.0,0.0,1.0"
    "Family_DINO:1.0,1.0,0.0,0.0"
    "Family_CLIP:0.0,0.0,1.0,1.0"
)

# --- 基本設定 ---

# 反復回数 (十分な学習時間を確保するために増加)
#ITERATIONS=2000
ITERATIONS=4000

# バッチサイズ (品質安定のため増加。エラーが出る場合は32に戻してください)
#AUGS_PER_STEP=16
AUGS_PER_STEP=32

NUM_GENERATIONS=1

LR=0.002
IMAGE_SIZE=224
NUM_REF_IMAGES=10

# --- [品質制御パラメータ (ここが重要)] ---

# 1. Augmentationスケール (GitHub: 0.08 = 8%までズーム, MyCode: 0.8 = 80%まで)
MIN_SCALE=0.08
MAX_SCALE=1.0
#MIN_SCALE=0.8

# 2. ノイズ設定 (GitHub: std=0.2, p=0.5 = 強いノイズ, MyCode: std=0.05)
NOISE_STD=0.2
NOISE_PROB=0.5
#NOISE_STD=0.05

# 3. TV Loss (GitHub: 0.0 = ぼかさない, MyCode: 0.00025 = 滑らかにする)
#WEIGHT_TV=0.0
WEIGHT_TV=0.00025

# 4. ピラミッド開始解像度 (GitHub: 1 = 1x1から開始, MyCode: 16)
PYRAMID_START_RES=4
#PYRAMID_START_RES=16

# 文字設定
OVERLAY_TEXT=""
TEXT_COLOR="red"
FONT_SCALE=0.5

# 並列実行設定
MAX_JOBS=2
NUM_GPUS=1

# ========================================================
# 4. 実行ループ
# ========================================================
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="$EXPERIMENT_ROOT_DIR/$TIMESTAMP"
mkdir -p "$BASE_OUTPUT_DIR"

echo "=========================================="
echo "Starting Experiment with Configurable Augs"
echo "Scale: $MIN_SCALE - $MAX_SCALE"
echo "Noise: std=$NOISE_STD (prob=$NOISE_PROB)"
echo "TV Loss: $WEIGHT_TV"
echo "Pyramid Start: $PYRAMID_START_RES"
echo "Output: $BASE_OUTPUT_DIR"
echo "=========================================="

for PROJ_DIMS in "${PROJ_DIM_LIST[@]}"; do
    BATCH_DIR="$BASE_OUTPUT_DIR/gen_data"
    mkdir -p "$BATCH_DIR"

    if [ -z "$TARGET_CLASSES" ]; then
        CLASS_LOOP=("ALL")
    else
        CLASS_LOOP=($TARGET_CLASSES)
    fi

    PIDS=()
    JOB_COUNTER=0

    for CLASS_ID in "${CLASS_LOOP[@]}"; do
        GPU_ID=$((JOB_COUNTER % NUM_GPUS))
        JOB_COUNTER=$((JOB_COUNTER + 1))
        
        if [ "$CLASS_ID" == "ALL" ]; then
            TARGET_ARG=""
            LOG_FILENAME="run_all.log"
        else
            TARGET_ARG="--target_classes $CLASS_ID"
            LOG_FILENAME="class_${CLASS_ID}.log"
        fi
        
        LOG_FILE_PATH="$BATCH_DIR/$LOG_FILENAME"

        if [ "$JOB_COUNTER" -le 1 ]; then
            REDIRECT=""
            echo "   [GPU $GPU_ID] Class: $CLASS_ID (Log visible)"
        else
            REDIRECT="> /dev/null 2>&1"
            echo "   [GPU $GPU_ID] Class: $CLASS_ID (Log hidden)"
        fi

        CMD="cd \"$SRC_DIR\" && CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_genimg.py \
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
            --num_generations $NUM_GENERATIONS \
            --overlay_text \"$OVERLAY_TEXT\" \
            --text_color \"$TEXT_COLOR\" \
            --font_scale $FONT_SCALE \
            --pyramid_start_res $PYRAMID_START_RES \
            --min_scale $MIN_SCALE \
            --max_scale $MAX_SCALE \
            --noise_std $NOISE_STD \
            --noise_prob $NOISE_PROB \
            --log_file \"$LOG_FILE_PATH\" $REDIRECT &"

        eval "$CMD"
        PIDS+=($!)

        if [ ${#PIDS[@]} -ge $MAX_JOBS ]; then
            echo "   >>> Max jobs ($MAX_JOBS) reached. Waiting..."
            for pid in "${PIDS[@]}"; do
                wait "$pid"
            done
            PIDS=()
        fi

    done

    if [ ${#PIDS[@]} -gt 0 ]; then
        echo "   >>> Waiting for remaining jobs..."
        for pid in "${PIDS[@]}"; do
            wait "$pid"
        done
        PIDS=()
    fi
done

echo "=========================================="
echo "All Experiments Completed."
echo "Check results at: $BASE_OUTPUT_DIR"
echo "=========================================="