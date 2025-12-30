import os
import glob
import json
import re
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import easyocr
from tqdm import tqdm

# =========================================================
# 設定・準備
# =========================================================

class Evaluator:
    def __init__(self, device):
        self.device = device
        print(f"Initializing Evaluator on {device}...")

        # 1. 構造・知覚評価用 (SSIM, LPIPS)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)

        # 2. Visual Score用 (ResNet-50)
        # ImageNet事前学習済みモデルを使用
        weights = models.ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=weights).to(device)
        self.resnet.eval()
        self.resnet_transform = weights.transforms()

        # 3. Text Score用 (EasyOCR)
        # GPUが使えるならgpu=True
        use_gpu_ocr = (device.type == 'cuda')
        self.reader = easyocr.Reader(['en'], gpu=use_gpu_ocr, verbose=False)

    def preprocess_image(self, img_path):
        """画像を読み込み、Tensorに変換する"""
        try:
            img = Image.open(img_path).convert('RGB')
            # 評価用に224x224にリサイズ
            img = img.resize((224, 224))
            # Tensor (0-1) [B, C, H, W]
            tensor = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
            return img, tensor
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

    def calc_visual_score(self, img_tensor, target_class_idx):
        """
        ResNet-50によるターゲットクラスの予測確率を算出
        target_class_idx: ImageNetのクラスID (int)
        """
        with torch.no_grad():
            processed = self.resnet_transform(img_tensor) # ResNet用正規化
            logits = self.resnet(processed)
            probs = F.softmax(logits, dim=1)
            
            # ターゲットクラスの確率を取得
            score = probs[0, target_class_idx].item()
            
            # 参考: Top-1クラスと確率も取得
            top1_prob, top1_idx = torch.max(probs, dim=1)
            
        return score, top1_idx.item(), top1_prob.item()

    def calc_text_score(self, img_np, target_text):
        """
        OCRによるテキスト検出スコア
        target_text: 検出したい文字列 (例: 'ipod')
        """
        if not target_text:
            return 0.0, ""

        # EasyOCRはnumpy arrayまたはファイルパスを受け取る
        # img_npは PIL Image から np.array(img) で変換したもの
        results = self.reader.readtext(img_np)
        
        max_score = 0.0
        detected_text = ""

        # 検出された全テキストの中から、ターゲットを含むものを探す
        target_clean = target_text.lower().strip()
        
        for bbox, text, conf in results:
            text_clean = text.lower().strip()
            # 完全一致または部分一致を評価
            # Levenshtein距離などを使うとより厳密だが、ここでは簡易的に包含判定
            if target_clean in text_clean or text_clean in target_clean:
                if conf > max_score:
                    max_score = conf
                    detected_text = text
            
            # ターゲットが見つからなくても、何かしら文字があればその最大スコアを記録しておく（オプション）
            # 今回は「特定の文字」へのバイアスを見たいので、ターゲット一致のみを評価
            
        return max_score, detected_text

    def calc_similarity(self, gen_tensor, ref_tensor):
        """SSIMとLPIPSを計算"""
        with torch.no_grad():
            ssim = self.ssim_metric(gen_tensor, ref_tensor).item()
            # LPIPSは入力範囲が[-1, 1]推奨の場合があるが、torchmetricsは[0,1]でも動く(normalize=True推奨)
            lpips = self.lpips_metric(gen_tensor, ref_tensor).item()
        return ssim, lpips

# =========================================================
# メイン処理
# =========================================================

def parse_rta_filename(filename):
    """
    RTA100のファイル名から label と text を抽出
    例: label=apple_text=ipod.png -> ('apple', 'ipod')
    """
    pattern = re.compile(r'label=(.+?)_text=(.+?)\.')
    match = pattern.search(filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Model LGM Results")
    parser.add_argument("--results_dir", type=str, required=True, help="Root directory of experiments (e.g. classification_results/...)")
    parser.add_argument("--data_root", type=str, required=True, help="Original RTA100 dataset root for reference images")
    parser.add_argument("--output_file", type=str, default="evaluation_summary.csv", help="Output CSV file path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = Evaluator(device)

    # 結果ディレクトリを走査
    # 構造: results_dir / experiment_name / class_id_classname / metrics.json
    
    # metrics.json を探して、そこから実験情報を復元する
    metric_files = glob.glob(os.path.join(args.results_dir, "**", "metrics.json"), recursive=True)
    
    records = []
    
    print(f"Found {len(metric_files)} experiments. Starting evaluation...")
    
    for mf in tqdm(metric_files):
        exp_dir = os.path.dirname(mf)
        
        # 1. 実験設定の読み込み
        with open(mf, 'r') as f:
            data = json.load(f)
            exp_args = data.get("args", {})
            metrics_hist = data.get("metrics", {}).get("loss_history", [])

        target_class_idx = exp_args.get("target_class")
        
        # 2. 画像パスの特定
        # ベストショットがあればそれを、なければ最終画像を使う
        best_img_path = os.path.join(exp_dir, "best_multi_model.png")
        final_img_path = os.path.join(exp_dir, "final_multi_model.png")
        
        target_img_path = best_img_path if os.path.exists(best_img_path) else final_img_path
        
        if not os.path.exists(target_img_path):
            continue

        # 3. 元画像の特定 (RTA100の場合)
        # ディレクトリ名からクラス名を推測する必要があるが、
        # RTA100はファイル名に情報が入っているため、オリジナルフォルダから検索する
        
        # ディレクトリ名: 948_apple_text_ipod みたいな形式を想定
        dir_name = os.path.basename(exp_dir)
        # ImageNet ID と クラス名文字列に分離 (簡易的)
        # format: "{id}_{sanitized_name}"
        parts = dir_name.split('_', 1)
        if len(parts) < 2: continue
        
        class_id_str = parts[0]
        sanitized_name = parts[1] # 例: apple_text_ipod
        
        # 元画像をデータセットフォルダから探す
        # sanitized_name は "label=apple_text=ipod" がサニタイズされたものかもしれない
        # ここは少しヒューリスティックだが、実験で使った target_class_idx をキーに探すのが確実
        # RTA100の場合、1クラス1画像前提なら、data_root内の全ファイルを舐めてマッチするものを探す
        
        ref_image_path = None
        target_text_label = ""
        visual_label = ""
        
        # オリジナル画像の検索 (metrics.jsonにファイルパスが残っていないため再検索)
        # 全ファイルを走査するのは重いので、キャッシュするか、あるいは実験時にパスを保存しておくのがベストだった
        # ここでは簡易的に glob で探す
        candidates = glob.glob(os.path.join(args.data_root, "*"))
        for c in candidates:
            # ファイル名から label と text をパース
            l_val, t_val = parse_rta_filename(os.path.basename(c))
            if l_val and t_val:
                # このファイルが、今回の実験対象かどうか判定
                # 簡易判定: sanitizeした結果がディレクトリ名に含まれているか
                # または、args.target_classes に入っていた名前と一致するか
                
                # ここでは、ディレクトリ名に含まれる文字列で判定 (弱連結)
                # 例: dir="950_orange", file="label=orange_text=ipod.png"
                # sanitized_name に l_val が含まれていればOKとする
                if l_val in sanitized_name:
                    ref_image_path = c
                    target_text_label = t_val
                    visual_label = l_val
                    break
        
        if not ref_image_path:
            # 元画像が見つからない場合はスキップまたはSSIMなしで続行
            # print(f"Warning: Reference image not found for {dir_name}")
            pass

        # ==========================
        # 評価実行
        # ==========================
        
        # 画像読み込み
        gen_img, gen_tensor = evaluator.preprocess_image(target_img_path)
        ref_img, ref_tensor = None, None
        if ref_image_path:
            ref_img, ref_tensor = evaluator.preprocess_image(ref_image_path)
        
        if gen_tensor is None: continue

        # Metric A: Visual Score (ResNet-50 Probability)
        # target_class_idx は ImageNet ID なのでそのまま使える
        vis_score, top1_idx, top1_prob = evaluator.calc_visual_score(gen_tensor, target_class_idx)
        
        # Metric B: Text Score (OCR Confidence)
        text_score = 0.0
        detected_text = ""
        if target_text_label:
            # EasyOCRにはnumpy配列を渡す
            import numpy as np
            gen_np = np.array(gen_img)
            text_score, detected_text = evaluator.calc_text_score(gen_np, target_text_label)

        # Metric C: SSIM & LPIPS
        ssim_val, lpips_val = None, None
        if ref_tensor is not None:
            ssim_val, lpips_val = evaluator.calc_similarity(gen_tensor, ref_tensor)

        # 実験条件の抽出 (重みなど)
        # exp_str (例: "Only_v1:1.0,0.0,0.0") を特定したいが、
        # argsにはリストで入っているので、ディレクトリ階層(exp_name)から逆引きする
        exp_name = os.path.basename(os.path.dirname(exp_dir)) # 親ディレクトリ名
        
        # レコード保存
        records.append({
            "exp_name": exp_name,
            "class_id": target_class_idx,
            "visual_label": visual_label,
            "text_label": target_text_label,
            
            # Metrics
            "visual_score": vis_score, # ResNet Prob
            "text_score": text_score,  # OCR Conf
            "ssim": ssim_val,
            "lpips": lpips_val,
            
            # Additional Info
            "detected_text": detected_text,
            "top1_class_id": top1_idx,
            "top1_prob": top1_prob,
            "image_path": target_img_path
        })

    # ==========================
    # 集計と保存
    # ==========================
    df = pd.DataFrame(records)
    df.to_csv(args.output_file, index=False)
    print(f"Evaluation completed. Saved to {args.output_file}")
    
    # 簡易レポート表示
    print("\n=== Summary by Experiment Type ===")
    if not df.empty:
        summary = df.groupby("exp_name")[["visual_score", "text_score", "ssim", "lpips"]].mean()
        print(summary)

if __name__ == "__main__":
    main()