# =========================
# main_genimg.py  (旧・安定版, 特徴保持なし)
# =========================
import os
import sys
import argparse
import json
import re
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision import datasets as tv_datasets
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from tqdm import tqdm
from datasets import load_dataset
from torchvision import models
from torch.utils.data import Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import easyocr

from optimized_model_utils import (
    EncoderClassifier,
    MultiModelGM,
    manage_model_allocation,
)

# ==============================================================================
# Custom Dataset Definitions (CUB-200)
# ==============================================================================
class CUB200(Dataset):
    def __init__(self, root, train=True, transform=None, loader=default_loader, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = loader
        self.train = train
        self.base_folder = os.path.join(self.root, "CUB_200_2011")

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self._load_metadata()

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.base_folder, "images.txt"), sep=" ",
                             names=["img_id", "filepath"])
        image_class_labels = pd.read_csv(os.path.join(self.base_folder, "image_class_labels.txt"),
                                         sep=" ", names=["img_id", "target"])
        train_test_split = pd.read_csv(os.path.join(self.base_folder, "train_test_split.txt"),
                                       sep=" ", names=["img_id", "is_training_img"])

        data = images.merge(image_class_labels, on="img_id")
        data = data.merge(train_test_split, on="img_id")

        classes_txt = pd.read_csv(os.path.join(self.base_folder, "classes.txt"), sep=" ",
                                  names=["class_id", "class_name"])
        self.classes = classes_txt["class_name"].apply(lambda x: x.split(".")[1].replace("_", " ")).tolist()

        if self.train:
            self.data = data[data.is_training_img == 1]
        else:
            self.data = data[data.is_training_img == 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.base_folder, "images", sample.filepath)
        target = sample.target - 1
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def _check_integrity(self):
        return os.path.isdir(os.path.join(self.root, "CUB_200_2011"))

    def _download(self):
        import tarfile
        url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
        filename = "CUB_200_2011.tgz"
        if not os.path.exists(os.path.join(self.root, filename)):
            print(f"Downloading {url}...")
            download_url(url, self.root, filename, None)

        print("Extracting...")
        tgz_path = os.path.join(self.root, filename)
        with tarfile.open(tgz_path, "r:gz") as tar:
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

            for member in tar.getmembers():
                member_path = os.path.join(self.root, member.name)
                if not is_within_directory(self.root, member_path):
                    raise RuntimeError(f"Unsafe path detected in tar file: {member.name}")
            tar.extractall(path=self.root)

# ==============================================================================
# Helpers
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def sanitize_dirname(name):
    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    return safe_name.strip("_")

def add_text_overlay(pil_image, text, font_scale=0.15, color=(255, 0, 0)):
    img = pil_image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    font_size = int(h * font_scale)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w = right - left
        text_h = bottom - top
    except Exception:
        text_w, text_h = draw.textsize(text, font=font)

    x = (w - text_w) // 2
    y = (h - text_h) // 2
    draw.text((x, y), text, fill=color, font=font)
    return img

class Logger:
    def __init__(self, log_path=None):
        self.log_path = log_path
        if log_path:
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self.file = open(log_path, "w", encoding="utf-8")
        else:
            self.file = None

    def log(self, message):
        tqdm.write(message)
        if self.file:
            self.file.write(message + "\n")
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()

# ==============================================================================
# Evaluator
# ==============================================================================
class Evaluator:
    def __init__(self, device, logger=None):
        self.device = device

        self.ssim_metric = None
        self.lpips_metric = None
        try:
            self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        except Exception as e:
            if logger:
                logger.log(f"[Warn] SSIM disabled: {e}")

        try:
            self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
        except Exception as e:
            if logger:
                logger.log(f"[Warn] LPIPS disabled: {e}")

        weights = models.ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=weights).to(device)
        self.resnet.eval()
        self.resnet_transform = weights.transforms()

        self.reader = None
        try:
            use_gpu_ocr = (device.type == "cuda")
            self.reader = easyocr.Reader(["en"], gpu=use_gpu_ocr, verbose=False)
        except Exception as e1:
            if logger:
                logger.log(f"[Warn] EasyOCR GPU init failed, fallback to CPU: {e1}")
            try:
                self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            except Exception as e2:
                if logger:
                    logger.log(f"[Warn] EasyOCR disabled: {e2}")
                self.reader = None

    def preprocess_tensor(self, img_tensor):
        return F.interpolate(img_tensor, size=(224, 224), mode="bilinear", align_corners=False)

    def calc_visual_score(self, img_tensor, target_class_idx):
        img_tensor = self.preprocess_tensor(img_tensor)
        with torch.no_grad():
            processed = self.resnet_transform(img_tensor)
            logits = self.resnet(processed)
            probs = F.softmax(logits, dim=1)
            score = probs[0, target_class_idx].item() if target_class_idx < 1000 else 0.0
            top1_prob, top1_idx = torch.max(probs, dim=1)
        return score, top1_idx.item(), top1_prob.item()

    def calc_text_score(self, img_tensor, target_text):
        if (not target_text) or (self.reader is None):
            return 0.0, ""

        img_cpu = img_tensor.detach().cpu().squeeze(0)
        img_pil = to_pil_image(img_cpu)
        img_np = np.array(img_pil)

        results = self.reader.readtext(img_np)
        max_score = 0.0
        detected_text = ""
        target_clean = target_text.lower().strip()

        for bbox, text, conf in results:
            text_clean = text.lower().strip()
            if target_clean in text_clean or text_clean in target_clean:
                if conf > max_score:
                    max_score = conf
                    detected_text = text
        return max_score, detected_text

    def calc_similarity(self, gen_tensor, ref_tensor):
        gen_resized = self.preprocess_tensor(gen_tensor)
        ref_resized = self.preprocess_tensor(ref_tensor)
        with torch.no_grad():
            ssim = self.ssim_metric(gen_resized, ref_resized).item() if self.ssim_metric is not None else float("nan")
            lpips = self.lpips_metric(gen_resized, ref_resized).item() if self.lpips_metric is not None else float("nan")
        return ssim, lpips

# ==============================================================================
# Args, Dataset
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Model LGM (Stable, No Cache)")

    parser.add_argument("--encoder_names", type=str, nargs="+", required=True)
    parser.add_argument("--projection_dims", type=int, nargs="+", default=[2048])
    parser.add_argument("--experiments", type=str, nargs="+", required=True)

    parser.add_argument("--output_dir", type=str, default="./lgm_results")
    parser.add_argument("--target_classes", type=str, nargs="+", default=[])
    parser.add_argument("--initial_image_path", type=str, default=None)
    parser.add_argument("--seed_noise_level", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", type=str, default="training_log.txt")

    parser.add_argument("--num_ref_images", type=int, default=10)
    parser.add_argument("--augs_per_step", type=int, default=32)
    parser.add_argument("--num_iterations", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--pyramid_start_res", type=int, default=16)
    parser.add_argument("--pyramid_grow_interval", type=int, default=400)
    parser.add_argument("--weight_grad", type=float, default=1.0)
    parser.add_argument("--weight_tv", type=float, default=0.00025)

    parser.add_argument("--num_generations", type=int, default=1)

    parser.add_argument("--overlay_text", type=str, default="")
    parser.add_argument("--text_color", type=str, default="red")
    parser.add_argument("--font_scale", type=float, default=0.15)

    parser.add_argument("--dataset_type", type=str, default="imagenet", choices=["imagenet", "food101", "cub200"])
    parser.add_argument("--data_root", type=str, default="./data")

    parser.add_argument("--min_scale", type=float, default=0.08)
    parser.add_argument("--max_scale", type=float, default=1.0)
    parser.add_argument("--noise_prob", type=float, default=0.5)
    parser.add_argument("--noise_std", type=float, default=0.2)

    parser.add_argument("--class_id_base", type=int, default=0, choices=[0, 1])
    parser.add_argument("--hf_streaming", action="store_true")

    parser.add_argument("--debug_grid_n", type=int, default=16)
    parser.add_argument("--debug_grid_nrow", type=int, default=4)

    return parser.parse_args()

def load_dataset_by_type(args, logger):
    if args.dataset_type == "imagenet":
        logger.log("Loading ImageNet (Hugging Face)...")
        dataset = load_dataset("imagenet-1k", split="validation", streaming=bool(args.hf_streaming))
        class_names = [str(i) for i in range(1000)] if args.hf_streaming else dataset.features["label"].names
        return dataset, class_names, "hf"

    if args.dataset_type == "food101":
        logger.log(f"Loading Food-101 (Torchvision), root={args.data_root} ...")
        try:
            dataset = tv_datasets.Food101(root=args.data_root, split="train", download=True)
        except Exception as e:
            logger.log(f"[Warn] Food101 download=True failed: {e}")
            logger.log("Retrying Food101 with download=False.")
            dataset = tv_datasets.Food101(root=args.data_root, split="train", download=False)
        return dataset, dataset.classes, "tv"

    if args.dataset_type == "cub200":
        logger.log("Loading CUB-200-2011 (Custom)...")
        dataset = CUB200(root=args.data_root, train=True, download=True)
        return dataset, dataset.classes, "tv"

    raise ValueError(f"Unknown dataset type: {args.dataset_type}")

def _get_label_array_for_tv_dataset(dataset):
    if hasattr(dataset, "_labels"):
        return np.asarray(dataset._labels)
    if hasattr(dataset, "data") and isinstance(dataset.data, pd.DataFrame) and ("target" in dataset.data.columns):
        return np.asarray(dataset.data["target"].values) - 1
    return None

def get_images_of_class_random_k(dataset, dataset_source_type, target_cls_idx, k, image_size, logger=None):
    collected = []

    if dataset_source_type == "hf":
        for item in dataset:
            if item["label"] == target_cls_idx:
                img = item["image"].convert("RGB").resize((image_size, image_size))
                collected.append(img)
                if len(collected) >= k:
                    break
        if logger:
            logger.log(f"[Debug] HF class={target_cls_idx}, collected={len(collected)} (sequential)")
        return collected

    labels = _get_label_array_for_tv_dataset(dataset)
    if labels is None:
        if logger:
            logger.log("[Warn] labels array not found, scanning full dataset (slow).")
        idxs = []
        for i in range(len(dataset)):
            _, y = dataset[i]
            if int(y) == int(target_cls_idx):
                idxs.append(i)
        idxs = np.asarray(idxs, dtype=np.int64)
    else:
        idxs = np.where(labels == target_cls_idx)[0]

    if logger:
        logger.log(f"[Debug] class={target_cls_idx}, class_population={len(idxs)}")

    if len(idxs) == 0:
        return []

    k_eff = min(int(k), int(len(idxs)))
    chosen = np.random.choice(idxs, size=k_eff, replace=False)

    for idx in chosen:
        img, _ = dataset[int(idx)]
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        img = img.convert("RGB").resize((image_size, image_size))
        collected.append(img)

    if logger:
        logger.log(f"[Debug] class={target_cls_idx}, sampled={len(collected)}")
    return collected

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger(args.log_file)
    logger.log(f"Using GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Using CPU")

    logger.log("Initializing Evaluator...")
    evaluator = Evaluator(device, logger=logger)

    dataset, dataset_class_names, dataset_source_type = load_dataset_by_type(args, logger)
    logger.log(f"Dataset Loaded: {args.dataset_type}, Classes: {len(dataset_class_names)}")

    if not args.target_classes:
        logger.log("No target classes provided, exiting.")
        sys.exit(0)

    target_ids_to_run = []
    for t in args.target_classes:
        try:
            cid = int(t) - args.class_id_base
            target_ids_to_run.append(cid)
        except Exception:
            logger.log(f"[Warn] Could not parse target class: {t}")

    models_list = []
    proj_dims = args.projection_dims
    if len(proj_dims) == 1:
        proj_dims = proj_dims * len(args.encoder_names)

    logger.log("Initializing Models...")
    for name, p_dim in zip(args.encoder_names, proj_dims):
        m = EncoderClassifier(name, freeze_encoder=True, num_classes=1000, projection_dim=p_dim)
        m.to(device)
        m.eval()
        models_list.append(m)

    total_steps = len(target_ids_to_run) * len(args.experiments) * args.num_generations * args.num_iterations
    global_pbar = tqdm(total=total_steps, unit="step", desc="Total Progress", position=0, dynamic_ncols=True)

    evaluation_results = []

    for target_cls in target_ids_to_run:
        if target_cls < 0 or target_cls >= len(dataset_class_names):
            logger.log(f"Skipping class ID {target_cls}, out of range.")
            continue

        class_name = dataset_class_names[target_cls]
        safe_cls_name = sanitize_dirname(class_name)
        logger.log(f"Preparing class: {target_cls} ({class_name})")

        clean_images_pil = get_images_of_class_random_k(
            dataset,
            dataset_source_type,
            target_cls,
            k=args.num_ref_images,
            image_size=args.image_size,
            logger=logger
        )
        if not clean_images_pil:
            logger.log(f"No images found for class {target_cls}.")
            continue

        overlay_text = args.overlay_text

        attacked_images_tensor = []
        for img in clean_images_pil:
            if overlay_text:
                img_with_text = add_text_overlay(img, overlay_text, args.font_scale, args.text_color)
            else:
                img_with_text = img
            attacked_images_tensor.append(to_tensor(img_with_text))

        real_images_pool = torch.stack(attacked_images_tensor).to(device)

        debug_dir = os.path.join(args.output_dir, "debug_inputs")
        os.makedirs(debug_dir, exist_ok=True)
        grid_n = min(args.debug_grid_n, real_images_pool.shape[0])
        save_image(
            real_images_pool[:grid_n].detach().cpu(),
            os.path.join(debug_dir, f"{args.dataset_type}_{target_cls}_{safe_cls_name}_ref_inputs_grid.png"),
            nrow=args.debug_grid_nrow
        )

        ref_clean_tensor = to_tensor(clean_images_pil[0]).unsqueeze(0).to(device)
        ref_attacked_tensor = real_images_pool[0].unsqueeze(0)

        clean_vis_score, _, _ = evaluator.calc_visual_score(ref_clean_tensor, target_cls)
        clean_text_score, _ = evaluator.calc_text_score(ref_clean_tensor, overlay_text)
        attacked_vis_score, _, _ = evaluator.calc_visual_score(ref_attacked_tensor, target_cls)
        attacked_text_score, _ = evaluator.calc_text_score(ref_attacked_tensor, overlay_text)

        # gen loop
        for gen_idx in range(args.num_generations):
            current_seed = args.seed + gen_idx
            set_seed(current_seed)

            # experiments loop, 旧版は共有しない.
            for exp_str in args.experiments:
                exp_name, weights_str = exp_str.split(":")
                weights = [float(w) for w in weights_str.split(",")]

                manage_model_allocation(models_list, weights, device)

                save_dir = os.path.join(args.output_dir, f"{args.dataset_type}_{exp_name}", f"{target_cls}_{safe_cls_name}")
                gen_subdir = os.path.join(save_dir, f"gen_{gen_idx:02d}")
                os.makedirs(gen_subdir, exist_ok=True)

                # 参照プールも expごとに保存(旧版の挙動)
                save_image(real_images_pool[:min(16, real_images_pool.shape[0])].detach().cpu(),
                           os.path.join(save_dir, "ref_pool.png"), nrow=4)

                lgm = MultiModelGM(
                    models=models_list,
                    model_weights=weights,
                    target_class=target_cls,
                    args=args,
                    device=device,
                    initial_image=None,
                )

                # 旧版, expごとに毎回 real feats を計算.
                lgm.precompute_real_features(real_images_pool)

                final_img, best_img, metrics = lgm.run(
                    real_images_pool,
                    gen_subdir,
                    dataset_class_names,
                    logger,
                    global_pbar,
                    gen_idx
                )

                target_img_tensor = best_img if best_img is not None else final_img
                target_img_tensor = target_img_tensor.to(device)

                save_image(target_img_tensor, os.path.join(gen_subdir, f"result_gen{gen_idx:02d}.png"))
                with open(os.path.join(gen_subdir, f"metrics_gen{gen_idx:02d}.json"), "w") as f:
                    json.dump({"args": vars(args), "metrics": metrics}, f, indent=2)
                save_image(target_img_tensor, os.path.join(save_dir, f"result_gen{gen_idx:02d}.png"))

                vis_score, top1_idx, top1_prob = evaluator.calc_visual_score(target_img_tensor, target_cls)
                text_score, detected_text = evaluator.calc_text_score(target_img_tensor, overlay_text)
                ssim_val, lpips_val = evaluator.calc_similarity(target_img_tensor, ref_clean_tensor)

                result_entry = {
                    "dataset": args.dataset_type,
                    "exp_name": exp_name,
                    "class_id": target_cls,
                    "class_name": class_name,
                    "gen_idx": gen_idx,
                    "overlay_text": overlay_text,
                    "clean_visual_score": clean_vis_score,
                    "clean_text_score": clean_text_score,
                    "attacked_visual_score": attacked_vis_score,
                    "attacked_text_score": attacked_text_score,
                    "result_visual_score": vis_score,
                    "result_text_score": text_score,
                    "ssim": ssim_val,
                    "lpips": lpips_val,
                    "detected_text": detected_text
                }
                evaluation_results.append(result_entry)
                logger.log(f"   [Eval][{exp_name}][Gen{gen_idx}] Vis:{vis_score:.4f}, Text:{text_score:.4f}")

    global_pbar.close()

    if evaluation_results:
        df = pd.DataFrame(evaluation_results)
        csv_path = os.path.join(args.output_dir, f"final_evaluation_{args.dataset_type}.csv")
        df.to_csv(csv_path, index=False)
        logger.log(f"All experiments finished. Saved to {csv_path}")
    else:
        logger.log("No experiments ran. Check class_id_base, or dataset class labels.")

    logger.close()
