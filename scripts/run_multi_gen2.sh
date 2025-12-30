#!/bin/bash
set -e

# ========================================================
# 1. 環境設定
# ========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
RESULTS_BASE_DIR="$PROJECT_DIR/classification_results"
# 出力先ディレクトリ名 (高速版として _fast を付与)
EXPERIMENT_ROOT_DIR="$RESULTS_BASE_DIR/gene_experiment_4models_fast"

mkdir -p "$EXPERIMENT_ROOT_DIR"
export PYTHONPATH="$SRC_DIR:$PROJECT_DIR:$PYTHONPATH"

# ========================================================
# 2. データセット設定
# ========================================================
DATASET_TYPE="imagenet"
DATA_ROOT="" 

# 【厳選8クラス】
# 1. ダチョウ(9)    : 生物・詳細
# 2. 金魚(1)        : 色彩・透明感
# 3. ハスキー(250)  : 毛並み・犬
# 4. 旅客機(404)    : 人工物・形状
# 5. 青リンゴ(948)  : シンプル物体・色
# 6. サッカー(805)  : 幾何学模様
# 7. シマウマ(340)  : [NEW] 強烈なテクスチャ
# 8. 火山(980)      : [NEW] 不定形・風景・爆発

# 【追加20クラス】
# 84(クジャク), 323(蝶), 937(ブロッコリー), 992(キノコ), 949(イチゴ)
# 107(クラゲ), 776(サックス), 850(テディベア), 966(ワイン), 920(信号機)
# 483(城), 839(吊り橋), 820(SL), 402(ギター), 817(スポーツカー)
# 130(フラミンゴ), 301(てんとう虫), 933(バーガー), 701(パラシュート), 999(ペーパー)
#598(蜂の巣) 682（オベリスク） 815（蜘蛛の巣） 973（珊瑚礁）

#TARGET_CLASSES="9 1 250 404 948 805 340 980"
#TARGET_CLASSES="340 776 980 999"
#TARGET_CLASSES="130 483 937 993"
#TARGET_CLASSES="850 920"
#TARGET_CLASSES="598 682 815 973"
TARGET_CLASSES="1 9 130 250 340 404 483 598 682 776 805 815 850 920 937 948 973 980 993 999"


# ========================================================
# 3. 実験パラメータ
# ========================================================

# [モデル構成] LAIONを除外した 計4モデル
ENCODER_NAMES=(
    "facebook/dino-vitb16"            # [0] DINO v1
    "facebook/dinov2-base"            # [1] DINO v2
    "openai/clip-vit-base-patch16"    # [2] OpenAI CLIP
    "google/siglip-base-patch16-224"  # [3] SigLIP
)

PROJ_DIM_LIST=("0 0 0 0")

# [実験設定] 10条件
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

# --- [高速化・並列化のための調整] ---
# A6000なら並列数を上げてもVRAMに余裕があるため、
# 1ステップあたりの負荷を少し下げて(32->16)、回転数を上げます。

ITERATIONS=5000       # 3000 -> 2000 (収束済みのため短縮)
#AUGS_PER_STEP=16      # 32 -> 16 (並列実行時のGPU負荷軽減と速度向上)
AUGS_PER_STEP=32
NUM_GENERATIONS=5    # 5枚作成

# 固定パラメータ
LR=0.002
IMAGE_SIZE=224
NUM_REF_IMAGES=10
#WEIGHT_TV=0.00025
WEIGHT_TV=0.00025

# [ピラミッド学習設定]
# 低解像度から開始して徐々に解像度を上げる設定 (Noneで無効化)
#PYRAMID_START_RES=16
PYRAMID_START_RES=4

# [Clean設定] 文字なし
OVERLAY_TEXT=""
TEXT_COLOR="red"
#FONT_SCALE=0.15
FONT_SCALE=0.5

# --- [並列設定の変更点] ---
MAX_JOBS=4    # 6クラスすべて同時実行 (待ち時間ゼロへ)
NUM_GPUS=2    # 使用するGPU枚数 (ログから2枚あると判断)

# ========================================================
# 4. 実行ループ
# ========================================================
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="$EXPERIMENT_ROOT_DIR/$TIMESTAMP"
mkdir -p "$BASE_OUTPUT_DIR"

echo "=========================================="
echo "Starting High-Speed Experiment"
echo "Target Classes: 6 types (Parallel)"
echo "Settings: $ITERATIONS iter x $NUM_GENERATIONS gen"
echo "Pyramid Start Res: $PYRAMID_START_RES"
echo "GPUs: $NUM_GPUS / Jobs: $MAX_JOBS"
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
        # GPU IDの割り当てロジック変更
        # ジョブ番号をGPU枚数で割った余りを使う (0, 1, 0, 1, 0, 1...)
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

        # ログ表示制御 (最初の1つだけ画面に出す)
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
            --log_file \"$LOG_FILE_PATH\" $REDIRECT &"

        eval "$CMD"
        PIDS+=($!)

        # MAX_JOBSに達したら、そのバッチが終わるのを待つ
        if [ ${#PIDS[@]} -ge $MAX_JOBS ]; then
            echo "   >>> Max jobs ($MAX_JOBS) launched. Processing all classes simultaneously..."
            for pid in "${PIDS[@]}"; do
                wait "$pid"
            done
            PIDS=()
            echo "   >>> Batch finished."
        fi

    done

    # 残りのジョブがあれば待機
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