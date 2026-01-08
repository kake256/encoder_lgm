#!/bin/bash
# run_bank.sh
# Feature Bank対応 + ダウンロードモード(キャッシュ活用)
# 目的: ImageNet全量(または一部)をダウンロード・キャッシュし、高速に蒸留を行う
set -e

# ========================================================
# 1. 環境設定
# ========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
RESULTS_BASE_DIR="$PROJECT_DIR/classification_results"

# 結果保存ディレクトリ
EXPERIMENT_ROOT_DIR="$RESULTS_BASE_DIR/gene_experiment_feature_bank"
mkdir -p "$EXPERIMENT_ROOT_DIR"
export PYTHONPATH="$SRC_DIR:$PROJECT_DIR:$PYTHONPATH"

# ========================================================
# 2. データセット設定
# ========================================================
DATASET_TYPE="imagenet-1k"
DATASET_SPLIT="train" 

# 【重要】データセットのキャッシュ保存先 (DATA_ROOT)
# 指定なし("")の場合、デフォルトの ~/.cache/huggingface/datasets に保存されます。
# ImageNetは巨大(150GB+)なため、容量に余裕のあるパスの指定を推奨します。
# 例: DATA_ROOT="/mnt/hdd/huggingface_cache"
DATA_ROOT=""

# DATA_ROOTが指定されている場合のみディレクトリ作成
if [ -n "$DATA_ROOT" ]; then
    mkdir -p "$DATA_ROOT"
fi

# 処理対象のクラスID (スペース区切り)
# Pythonスクリプトにリストとして一括で渡します
TARGET_CLASSES="1 9 130 250 340 404 483 598 682 776 805 815 850 920 937 948 973 980 993 999"

# ========================================================
# 3. Feature Bank & 最適化パラメータ
# ========================================================
# [USE_REAL] 実画像の参照枚数 (Feature Bankの元ネタ)
#   0  : クラス内の全画像を使用 (推奨: 約1300枚。最も精度が高い)
#   N  : 先頭からN枚のみ使用 (例: 50)。ダウンロードモードなら0でもOKですが、初回は時間がかかります。
USE_REAL=200

# [REAL_AUG] 実画像1枚あたりの拡張数 (Feature Bankのサイズ決定)
#   例: 1300枚 x 10倍 = 13,000個の特徴量をキャッシュ
#   10〜20程度が推奨。多いほどメモリを使うが分布が密になる。
REAL_AUG=20

# [SYN_AUG] 1ステップあたりの比較枚数 (バッチサイズ)
#   生成画像を何通りに拡張して比較するか。
SYN_AUG=32

# [ITERATIONS] 最適化のループ回数
ITERATIONS=2500

# [NUM_GENERATIONS] 1クラスあたりの画像生成枚数 (★追加項目)
#   1  : 1枚だけ生成
#   5  : シードを変えて5枚生成 (実験の信頼性向上のため複数推奨)
NUM_GENERATIONS=1

# --- Augmentation (データ拡張) の強度設定 ---
MIN_SCALE=0.08      # RandomResizedCropの最小スケール
MAX_SCALE=1.0       # RandomResizedCropの最大スケール
NOISE_STD=0.2       # ガウシアンノイズの標準偏差
NOISE_PROB=0.5      # ノイズを加える確率

# --- その他の学習パラメータ ---
WEIGHT_TV=0.00025       # TV Loss (滑らかさ) の重み

# --- ピラミッド解像度制御 (画像を細かくしていくペース) ---
PYRAMID_START_RES=16      # 開始解像度 (16x16)
PYRAMID_GROW_INTERVAL=400 # 何イテレーションごとに解像度を上げるか

# --- 文字攻撃 (Text Overlay) 設定 ---
# 空文字 "" なら適用なし。指定すると実画像に文字を合成してから特徴量を計算する。
OVERLAY_TEXT=""
TEXT_COLOR="red"
FONT_SCALE=0.15

# OCR機能の制御 (文字攻撃をしない場合は無効化して高速化・メモリ節約)
DISABLE_OCR_FLAG=""
if [ -z "$OVERLAY_TEXT" ]; then
    DISABLE_OCR_FLAG="--disable_ocr"
fi

# ========================================================
# 4. モデル設定
# ========================================================
# 使用するモデルのリスト (Hugging Face ID)
ENCODER_NAMES=(
  "facebook/dino-vitb16"           # Index 0: DINOv1
  "facebook/dinov2-base"           # Index 1: DINOv2
  "openai/clip-vit-base-patch16"   # Index 2: CLIP
  "google/siglip-base-patch16-224" # Index 3: SigLIP
)

# プロジェクション層の次元 (基本は0=なし)
PROJ_DIM_LIST=("0 0 0 0")

# 実験設定リスト
# 書式: "実験名:重み0,重み1,重み2,重み3"
EXPERIMENTS=(
  # --- 単体モデル (基準用) ---
  "Only_V1:1.0,0.0,0.0,0.0"       # DINOv1のみ
  "Only_V2:0.0,1.0,0.0,0.0"       # DINOv2のみ
  "Only_CLIP:0.0,0.0,1.0,0.0"     # CLIPのみ
  "Only_SigLIP:0.0,0.0,0.0,1.0"   # SigLIPのみ

  # --- マルチモデル (ハイブリッド) ---
  "Hybrid_V1_V2:1.0,1.0,0.0,0.0"         # DINO同士
  "Hybrid_CLIP_SigLIP:0.0,0.0,1.0,1.0"   # 言語系同士
)

# ========================================================
# 5. 実行
# ========================================================
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="$EXPERIMENT_ROOT_DIR/$TIMESTAMP"
CACHE_DIR="$EXPERIMENT_ROOT_DIR/feature_cache" # 特徴量バンクのキャッシュ

mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

echo "=========================================="
echo "Starting Feature Bank Optimization (Batch Mode)"
echo "  - Cache Dir (Dataset): ${DATA_ROOT:-Default(~/.cache/huggingface)}"
echo "  - Cache Dir (Features): $CACHE_DIR"
echo "  - Output Dir: $BASE_OUTPUT_DIR"
echo "  - Target Classes: $TARGET_CLASSES"
echo "  - Generations per Class: $NUM_GENERATIONS"
echo "=========================================="

# GPU ID指定 (シングルGPU想定)
GPU_ID=0

for PROJ_DIMS in "${PROJ_DIM_LIST[@]}"; do
  BATCH_DIR="$BASE_OUTPUT_DIR/gen_data"
  mkdir -p "$BATCH_DIR"

  LOG_FILE_PATH="$BATCH_DIR/training_log.txt"

  # ★変更: --num_generations を追加
  CMD="cd \"$SRC_DIR\" && CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_feature_bank.py \
    --encoder_names ${ENCODER_NAMES[@]} \
    --projection_dims $PROJ_DIMS \
    --experiments ${EXPERIMENTS[@]} \
    --target_classes $TARGET_CLASSES \
    --dataset_type \"$DATASET_TYPE\" \
    --dataset_split \"$DATASET_SPLIT\" \
    --data_root \"$DATA_ROOT\" \
    --output_dir \"$BATCH_DIR\" \
    --cache_dir \"$CACHE_DIR\" \
    --num_iterations $ITERATIONS \
    --num_generations $NUM_GENERATIONS \
    --use_real $USE_REAL \
    --real_aug $REAL_AUG \
    --syn_aug $SYN_AUG \
    --weight_tv $WEIGHT_TV \
    --overlay_text \"$OVERLAY_TEXT\" \
    --text_color \"$TEXT_COLOR\" \
    --font_scale $FONT_SCALE \
    --pyramid_start_res $PYRAMID_START_RES \
    --pyramid_grow_interval $PYRAMID_GROW_INTERVAL \
    --min_scale $MIN_SCALE \
    --max_scale $MAX_SCALE \
    --noise_std $NOISE_STD \
    --noise_prob $NOISE_PROB \
    $DISABLE_OCR_FLAG \
    --log_file \"$LOG_FILE_PATH\""

  echo "Running optimization..."
  eval "$CMD"
done

echo "Done. Results saved to: $BASE_OUTPUT_DIR"