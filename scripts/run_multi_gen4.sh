#!/bin/bash
# ==============================================================================
# 最適化されたバッチ処理による画像生成実験実行スクリプト
# 役割: 実験パラメータを一元管理し、Pythonプログラム(main_genimg2.py)を起動する
# ==============================================================================
set -e

# ========================================================
# 1. 環境設定 (ディレクトリパスなど)
# ========================================================
# スクリプトの場所を基準に、プロジェクトのルートやソースコードのパスを自動取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
RESULTS_BASE_DIR="$PROJECT_DIR/classification_results"

# 実験結果の保存先ルートディレクトリ
EXPERIMENT_ROOT_DIR="$RESULTS_BASE_DIR/gene_experiment_batched_high_quality"

# ディレクトリ作成とPYTHONPATHの設定（モジュール読み込みのため）
mkdir -p "$EXPERIMENT_ROOT_DIR"
export PYTHONPATH="$SRC_DIR:$PROJECT_DIR:$PYTHONPATH"

# ========================================================
# 2. データセット設定
# ========================================================
DATASET_TYPE="imagenet"
DATA_ROOT="" 

# --------------------------------------------------------
# 【ターゲットクラス設定】
# 以下のリストから、生成したいクラスIDをスペース区切りで指定してください。
# この実験セットでは、質感・形状・色彩の多様性を評価するために
# 以下の20クラスが推奨されています。
# --------------------------------------------------------

# --- [A. 生物・自然 (Bio/Nature)] ---
# 1   : Goldfish (金魚)           - 【色彩・透明感】赤・オレンジの鮮やかな発色と、ヒレの透け感の再現
# 9   : Ostrich (ダチョウ)        - 【細部・生物】羽毛の細かいディテールと、特徴的な体の構造
# 130 : Flamingo (フラミンゴ)     - 【色彩・群れ】独特なピンク色、細い脚、複数体集まった構図
# 250 : Siberian husky (ハスキー) - 【毛並み・犬】白黒の毛並みの質感、犬の顔の構造的整合性
# 340 : Zebra (シマウマ)          - 【高周波・模様】白黒の強いコントラストとストライプ模様の再現
# 815 : Spider web (クモの巣)     - 【極細線】背景に溶け込みやすい、非常に細い線の再構築能力
# 973 : Coral reef (サンゴ礁)     - 【複雑性・色彩】ランダムで複雑な形状と、多色な色彩情報
# 980 : Volcano (火山)            - 【不定形・煙】噴煙や溶岩など、輪郭が曖昧なオブジェクトの表現
# 993 : Daisy (ヒナギク)          - 【植物・繰り返し】花弁の重なりや、中心部の粒状感

# --- [B. 人工物・構造物 (Artifacts/Structures)] ---
# 404 : Airliner (旅客機)         - 【滑らかさ・空】金属の光沢感、流線型のボディ、青空背景
# 483 : Castle (城)               - 【建築・複雑】石積み、塔、窓などの人工的な幾何学構造
# 682 : Obelisk (オベリスク)      - 【シンプル・直線】空に向かう単純な直線構造と石の質感
# 776 : Saxophone (サックス)      - 【金属光沢】複雑な曲面を持つ真鍮（金管）の反射・光沢
# 920 : Traffic light (信号機)    - 【発光・人工色】赤・黄・青の人工的な発色と、無機質な形状

# --- [C. 質感・テクスチャ (Textures/Materials)] ---
# 598 : Honeycomb (蜂の巣)        - 【幾何学パターン】規則正しく並ぶ六角形パターンの再現
# 805 : Soccer ball (サッカー球)  - 【幾何学・球体】球体の立体感と、五角形・六角形の模様
# 850 : Teddy bear (テディベア)   - 【毛の質感】動物の毛とは異なる、ぬいぐるみのモコモコした質感
# 937 : Broccoli (ブロッコリー)   - 【粒状感】蕾の集合体による、細かくザラザラした緑色の質感
# 948 : Granny Smith (青リンゴ)   - 【シンプル・色】滑らかな表面、単色のグラデーション、球体
# 999 : Toilet tissue (ペーパー)  - 【白単色・陰影】色彩情報が乏しい「白」の中での、柔らかな質感表現

# --------------------------------------------------------
# ▼▼ 実行するクラスをここで設定してください ▼▼
# --------------------------------------------------------

# [設定例1] 動作確認用（2クラスのみ）
TARGET_CLASSES="1 9 130 340"

# [設定例2] 質感重視セット
# TARGET_CLASSES="340 598 776 815 937 999"

# [設定例3] フルセット（全20種）
# TARGET_CLASSES="1 9 130 250 340 404 483 598 682 776 805 815 850 920 937 948 973 980 993 999"

# ========================================================
# 3. 実験パラメータ (モデルと重み付け)
# ========================================================

# 使用するエンコーダーモデルのリスト (HuggingFace / timm)
ENCODER_NAMES=(
    "facebook/dino-vitb16"           # Model 0: DINO v1
    "facebook/dinov2-base"           # Model 1: DINO v2
    "openai/clip-vit-base-patch16"   # Model 2: CLIP
    "google/siglip-base-patch16-224" # Model 3: SigLIP
)

# 各モデルの射影次元 (0の場合はデフォルトの次元を使用)
PROJ_DIMS="0 0 0 0"

# 実験設定リスト
# 書式: "実験名:モデル0の重み,モデル1の重み,モデル2の重み,モデル3の重み"
EXPERIMENTS=(
    "Only_V1:1.0,0.0,0.0,0.0"          # DINO v1のみ使用
    "Only_V2:0.0,1.0,0.0,0.0"          # DINO v2のみ使用
    "Only_CLIP:0.0,0.0,1.0,0.0"        # CLIPのみ使用
    "Only_SigLIP:0.0,0.0,0.0,1.0"      # SigLIPのみ使用
    "Hybrid_V1_CLIP:1.0,0.0,1.0,0.0"   # DINO v1 + CLIP
    "Hybrid_V1_SigLIP:1.0,0.0,0.0,1.0" # DINO v1 + SigLIP
    "Hybrid_V2_CLIP:0.0,1.0,1.0,0.0"   # DINO v2 + CLIP
    "Hybrid_V2_SigLIP:0.0,1.0,0.0,1.0" # DINO v2 + SigLIP
    "Family_DINO:1.0,1.0,0.0,0.0"      # DINO系のみ
    "Family_CLIP:0.0,0.0,1.0,1.0"      # CLIP系のみ
)

# ========================================================
# GPU設定 & バッチサイズ制御 (環境に合わせてスイッチ)
# ========================================================

# [設定箇所] 使用するGPUの枚数
# 2 => A6000 x 2 (GPU 0, 1を使用)
# 1 => Strong GPU x 1 (GPU 0を使用)
NUM_GPUS=1

if [ "$NUM_GPUS" -eq 2 ]; then
    GPU_ARG="0,1"
    # GPUが2枚ある場合、負荷分散されるためバッチサイズ(同時生成クラス数)を維持/拡大可能
    BATCH_SIZE=32
else
    GPU_ARG="0"
    # 強力なGPU1枚の場合の設定。VRAM容量に応じて調整してください(32〜64推奨)
    BATCH_SIZE=32
fi

# ========================================================
# ハイパーパラメータ (画質と学習速度の制御)
# ========================================================

# 最適化の反復回数 (多いほど高精細になるが時間がかかる。2000-5000推奨)
ITERATIONS=4000

# 1ステップあたりのAugmentation回数 (勾配の質を決める重要な値)
# メモリが許す限り大きくする (例: 32, 64, 128)
AUGS_PER_STEP=64

# [追加] 途中経過画像を保存するインターバル (ステップ数)
# デフォルト: 500 (例: 4000反復なら8回保存される)
SAVE_INTERVAL=500

# 学習率 (大きくしすぎると画像が崩壊する)
LR=0.002

# 生成画像のサイズ
IMAGE_SIZE=224

# 参照画像の枚数 (多いほどターゲットの特徴を平均的に捉えられる)
NUM_REF_IMAGES=10

# TV Lossの重み (画像のザラつきを抑え、滑らかにする効果)
# 0.0だとシャープだがノイズが増える。0.0001~0.0005推奨
WEIGHT_TV=0.00025

# ピラミッド生成の開始解像度 (4x4から開始して徐々に大きくする)
PYRAMID_START_RES=4

# --- Augmentation詳細設定 ---
# 画像を切り抜くスケール範囲 (0.08=8% 〜 1.0=100%)
MIN_SCALE=0.08
MAX_SCALE=1.0

# ガウシアンノイズの設定 (過学習を防ぎ、頑健な特徴を得るため)
NOISE_STD=0.2   # ノイズの強さ
NOISE_PROB=0.5  # ノイズを加える確率

# --- 文字オーバーレイ設定 (敵対的攻撃実験用) ---
OVERLAY_TEXT=""   # 画像に文字を書き込む場合ここに記述 (例: "ipod")
TEXT_COLOR="red"
FONT_SCALE=0.5

# ========================================================
# 4. 実行処理
# ========================================================
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="$EXPERIMENT_ROOT_DIR/$TIMESTAMP"
mkdir -p "$BASE_OUTPUT_DIR"

echo "=========================================="
echo "Starting Optimized Batched Experiment"
echo "GPUs Used      : $GPU_ARG"
echo "Target Classes : $TARGET_CLASSES"
echo "Output Dir     : $BASE_OUTPUT_DIR"
echo "=========================================="

# Pythonスクリプトの呼び出し
# 以前のループ処理は廃止し、Python内部でGPU分散・バッチ処理を行います
cd "$SRC_DIR" && python3 main_genimg2.py \
    --encoder_names ${ENCODER_NAMES[@]} \
    --projection_dims $PROJ_DIMS \
    --experiments ${EXPERIMENTS[@]} \
    --target_classes $TARGET_CLASSES \
    --gpus "$GPU_ARG" \
    --dataset_type "$DATASET_TYPE" \
    --data_root "$DATA_ROOT" \
    --output_dir "$BASE_OUTPUT_DIR" \
    --num_iterations $ITERATIONS \
    --lr $LR \
    --image_size $IMAGE_SIZE \
    --num_ref_images $NUM_REF_IMAGES \
    --augs_per_step $AUGS_PER_STEP \
    --weight_tv $WEIGHT_TV \
    --batch_size_gen $BATCH_SIZE \
    --overlay_text "$OVERLAY_TEXT" \
    --text_color "$TEXT_COLOR" \
    --font_scale $FONT_SCALE \
    --pyramid_start_res $PYRAMID_START_RES \
    --min_scale $MIN_SCALE \
    --max_scale $MAX_SCALE \
    --noise_std $NOISE_STD \
    --noise_prob $NOISE_PROB \
    --save_interval $SAVE_INTERVAL \
    --log_file "$BASE_OUTPUT_DIR/training.log"

echo "=========================================="
echo "All Experiments Completed."
echo "Check results at: $BASE_OUTPUT_DIR"
echo "=========================================="