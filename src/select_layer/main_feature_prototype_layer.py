# =========================
# src/main_feature_prototype_layer.py
# (No cache version, switchable single/multi real-image DA, selectable feature layer,
#  with prototype feature alignment)
# =========================
import os
import sys
import argparse
import json
import re
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from datasets import load_dataset_builder
import pandas as pd
from torchvision import models
from torchvision.datasets import ImageFolder
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import easyocr
import urllib.request

from model_utils_prototype_layer import (
    EncoderClassifier,
    MultiModelGM,
    FeatureBankSystem,
    manage_model_allocation,
)


class Evaluator:
    def __init__(self, device, use_ocr=True, disable_all=False):
        self.device = device
        self.use_ocr = use_ocr
        self.disable_all = disable_all

        if self.disable_all:
            self.resnet = None
            self.reader = None
            return

        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)

        weights = models.ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=weights).to(device)
        self.resnet.eval()
        self.resnet_transform = weights.transforms()

        if self.use_ocr:
            use_gpu_ocr = (device.type == "cuda")
            self.reader = easyocr.Reader(["en"], gpu=use_gpu_ocr, verbose=False)
        else:
            self.reader = None

    def preprocess_tensor(self, img_tensor):
        return F.interpolate(img_tensor, size=(224, 224), mode="bilinear", align_corners=False)

    def calc_visual_score(self, img_tensor, target_class_idx):
        if self.disable_all:
            return 0.0, 0, 0.0

        img_tensor = self.preprocess_tensor(img_tensor)
        with torch.no_grad():
            processed = self.resnet_transform(img_tensor)
            logits = self.resnet(processed)
            probs = F.softmax(logits, dim=1)
            if target_class_idx < 1000:
                score = probs[0, target_class_idx].item()
            else:
                score = 0.0
            top1_prob, top1_idx = torch.max(probs, dim=1)
        return score, top1_idx.item(), top1_prob.item()

    def calc_text_score(self, img_tensor, target_text):
        if self.disable_all or not self.use_ocr or not target_text:
            return 0.0, ""

        img_cpu = img_tensor.detach().cpu().squeeze(0)
        img_pil = to_pil_image(img_cpu)
        img_np = np.array(img_pil)
        results = self.reader.readtext(img_np)

        max_score = 0.0
        detected_text = ""
        target_clean = target_text.lower().strip()

        for _bbox, text, conf in results:
            text_clean = text.lower().strip()
            if target_clean in text_clean or text_clean in target_clean:
                if conf > max_score:
                    max_score = conf
                    detected_text = text

        return max_score, detected_text

    def calc_similarity(self, gen_tensor, ref_tensor):
        if self.disable_all or ref_tensor is None:
            return 0.0, 0.0

        if ref_tensor.device != self.device:
            ref_tensor = ref_tensor.to(self.device)

        gen_resized = self.preprocess_tensor(gen_tensor)
        ref_resized = self.preprocess_tensor(ref_tensor)

        with torch.no_grad():
            ssim = self.ssim_metric(gen_resized, ref_resized).item()
            lpips = self.lpips_metric(gen_resized, ref_resized).item()

        return ssim, lpips


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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def sanitize_dirname(name):
    first_name = name.split(",")[0].strip()
    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", first_name)
    return safe_name.strip("_")


def add_text_overlay(pil_image, text, font_scale=0.15, color="red"):
    if text is None or str(text).strip() == "":
        return pil_image

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


def collect_target_images(dataset, target_class_ids, images_per_class, max_scan_global, image_size, logger):
    collected_images = {cls_id: [] for cls_id in target_class_ids}
    needs_collection = {cls_id: True for cls_id in target_class_ids}

    total_samples = len(dataset) if hasattr(dataset, "__len__") else max_scan_global
    actual_scan_limit = min(total_samples, max_scan_global)

    logger.log(f"Starting Dataset Scan (Total: {total_samples} images)...")
    logger.log(f"Looking for classes: {target_class_ids}")

    with tqdm(total=actual_scan_limit, desc="Scanning", unit="img") as pbar:
        for i, item in enumerate(dataset):
            if i >= actual_scan_limit:
                break
            pbar.update(1)

            if isinstance(item, tuple):
                img_raw, label_idx = item
            else:
                try:
                    label_idx = int(item["label"])
                    img_raw = item["image"]
                except Exception:
                    continue

            if label_idx in collected_images and needs_collection[label_idx]:
                img = img_raw.convert("RGB").resize((image_size, image_size))
                collected_images[label_idx].append(img)
                if len(collected_images[label_idx]) >= images_per_class:
                    needs_collection[label_idx] = False

            if not any(needs_collection.values()):
                logger.log("All target classes collected.")
                break

    return collected_images


def load_class_names_offline(cache_dir, dataset_type="imagenet-1k"):
    if dataset_type.lower() == "flowers102":
        return [f"Flower_{i}" for i in range(102)]

    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", dataset_type)
    json_path = os.path.join(cache_dir, f"{safe_name}_labels.json")

    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception:
            pass

    if "imagenet-100" in dataset_type:
        try:
            ds_builder = load_dataset_builder(dataset_type, trust_remote_code=True)
            names = ds_builder.info.features["label"].names
            with open(json_path, "w") as f:
                json.dump(names, f)
            return names
        except Exception as e:
            print(f"[Warning] Failed to fetch ImageNet-100 labels: {e}")

    try:
        ds = load_dataset(dataset_type, split="train", streaming=True, trust_remote_code=True)
        if hasattr(ds, "features") and "label" in ds.features:
            names = ds.features["label"].names
            with open(json_path, "w") as f:
                json.dump(names, f)
            return names
    except Exception:
        pass

    if "imagenet" in dataset_type.lower() and "100" not in dataset_type:
        url = "https://huggingface.co/datasets/huggingface/label-files/resolve/main/imagenet-1k-id2label.json"
        try:
            with urllib.request.urlopen(url) as response:
                id2label = json.loads(response.read().decode())
                names = [id2label[str(i)] for i in range(1000)]
                with open(json_path, "w") as f:
                    json.dump(names, f)
                return names
        except Exception:
            pass

    default_count = 100 if "100" in dataset_type else 1000
    return [str(i) for i in range(default_count)]


def append_completion_log(output_dir, class_id, class_name, exp_name, gen_idx):
    import datetime

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "completion_log.csv")

    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Timestamp,Class_ID,Class_Name,Experiment,Gen_Idx,Status\n")

    with open(log_path, "a", encoding="utf-8") as f:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts},{class_id},{class_name},{exp_name},{gen_idx},DONE\n")


def prepare_real_image_pool_tensor(pil_images_list, image_size, overlay_text, font_scale, text_color):
    tensor_list = []
    for p_img in pil_images_list:
        p_img = p_img.convert("RGB").resize((image_size, image_size))
        img_with_text = add_text_overlay(p_img, overlay_text, font_scale, text_color)
        tensor_list.append(to_tensor(img_with_text))

    if len(tensor_list) == 0:
        return torch.empty(0, 3, image_size, image_size)

    return torch.stack(tensor_list, dim=0).cpu()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--encoder_names", type=str, nargs="+", required=True)
    parser.add_argument("--projection_dims", type=int, nargs="+", default=[2048])
    parser.add_argument("--experiments", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="./lgm_results")
    parser.add_argument("--target_classes", type=str, nargs="+", default=[])
    parser.add_argument("--initial_image_path", type=str, default=None)
    parser.add_argument("--seed_noise_level", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", type=str, default="training_log.txt")
    parser.add_argument("--num_iterations", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--batch_opt_size", type=int, default=10, help="Number of classes to optimize simultaneously")
    parser.add_argument("--pyramid_start_res", type=int, default=16)
    parser.add_argument("--pyramid_grow_interval", type=int, default=400)
    parser.add_argument("--weight_grad", type=float, default=1.0)
    parser.add_argument("--weight_proto", type=float, default=0.1)
    parser.add_argument("--weight_tv", type=float, default=0.00025)
    parser.add_argument("--use_real", type=int, default=0)
    parser.add_argument("--max_scan_limit", type=int, default=1300000)
    parser.add_argument("--real_aug", type=int, default=10, help="Backward compatibility only. Not used directly.")
    parser.add_argument("--syn_aug", type=int, default=32)
    parser.add_argument("--cache_dir", type=str, default="./feature_cache")
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--overlay_text", type=str, default="ipod")
    parser.add_argument("--text_color", type=str, default="red")
    parser.add_argument("--font_scale", type=float, default=0.15)
    parser.add_argument("--disable_ocr", action="store_true")
    parser.add_argument("--dataset_type", type=str, default="imagenet-1k")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--min_scale", type=float, default=0.08)
    parser.add_argument("--max_scale", type=float, default=1.0)
    parser.add_argument("--noise_prob", type=float, default=0.5)
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--streaming", action="store_true")

    parser.add_argument("--only_bank_creation", action="store_true", help="Deprecated. Kept only for compatibility.")
    parser.add_argument("--no_evaluation", action="store_true", help="Skip real image loading and SSIM/LPIPS evaluation")

    # 追加 1. 1枚DA / 複数枚DA の切替
    parser.add_argument(
        "--real_sampling_mode",
        type=str,
        default="single",
        choices=["single", "multi"],
        help="single: 1枚選んでDA, multi: 複数枚選んでDA"
    )
    parser.add_argument(
        "--real_images_per_step",
        type=int,
        default=4,
        help="multi モード時に各 step で選ぶ実画像枚数"
    )
    parser.add_argument(
        "--real_aug_per_image",
        type=int,
        default=8,
        help="選んだ各実画像に対するDA回数"
    )

    # 追加 2. 特徴抽出層の指定
    parser.add_argument(
        "--feature_layer",
        type=int,
        default=-1,
        help="-1: 最終層, 0以上または負の値で hidden_states の層を指定"
    )
    parser.add_argument(
        "--feature_source",
        type=str,
        default="cls",
        choices=["pooler", "cls", "mean"],
        help="特徴抽出方法. 中間層使用時の pooler は cls 相当に寄せる"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger(args.log_file)
    logger.log(f"Using GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU")
    logger.log("Feature cache is disabled in this version.")
    logger.log(
        f"real_sampling_mode={args.real_sampling_mode}, "
        f"real_images_per_step={args.real_images_per_step}, "
        f"real_aug_per_image={args.real_aug_per_image}, "
        f"feature_layer={args.feature_layer}, "
        f"feature_source={args.feature_source}, "
        f"weight_proto={args.weight_proto}"
    )

    # 構造維持のため残すが, cache は使わない
    bank_system = FeatureBankSystem(args, device, args.cache_dir)

    if args.only_bank_creation:
        logger.log("[Info] --only_bank_creation is deprecated in this no-cache version. Exiting.")
        logger.close()
        sys.exit(0)

    target_ids_to_run = [int(t) for t in args.target_classes] if args.target_classes else []

    should_load_dataset = True
    if args.no_evaluation:
        logger.log("Evaluation metrics are disabled, but dataset loading remains enabled for real-image sampling.")

    collected_data_map = {}
    dataset_class_names = []

    if should_load_dataset:
        hf_cache_dir = args.data_root if args.data_root else None
        if hf_cache_dir:
            os.makedirs(hf_cache_dir, exist_ok=True)

        if args.dataset_type == "imagefolder":
            logger.log(f"Loading local dataset from: {args.data_root}")
            try:
                dataset = ImageFolder(args.data_root)
                dataset_class_names = dataset.classes
            except Exception as e:
                logger.log(f"Error loading ImageFolder: {e}")
                sys.exit(1)

        elif args.dataset_type.lower() == "flowers102":
            logger.log("Loading torchvision Flowers102 dataset (All splits combined)...")
            try:
                from torchvision.datasets import Flowers102
                from torch.utils.data import ConcatDataset

                ds_train = Flowers102(root=hf_cache_dir or "./data", split="train", download=True)
                ds_val = Flowers102(root=hf_cache_dir or "./data", split="val", download=True)
                ds_test = Flowers102(root=hf_cache_dir or "./data", split="test", download=True)
                dataset = ConcatDataset([ds_train, ds_val, ds_test])
                dataset_class_names = [f"Flower_{i}" for i in range(102)]
            except Exception as e:
                logger.log(f"Error loading Flowers102: {e}")
                sys.exit(1)

        else:
            try:
                dataset = load_dataset(
                    args.dataset_type,
                    split=args.dataset_split,
                    streaming=False,
                    trust_remote_code=True,
                    cache_dir=hf_cache_dir
                )
                try:
                    dataset_class_names = dataset.features["label"].names
                except Exception:
                    pass
            except Exception as e:
                logger.log(f"Error loading dataset: {e}")
                sys.exit(1)

        if dataset_class_names:
            safe_name = re.sub(r"[^a-zA-Z0-9]", "_", args.dataset_type)
            lbl_path = os.path.join(args.cache_dir, f"{safe_name}_labels.json")
            os.makedirs(args.cache_dir, exist_ok=True)
            with open(lbl_path, "w") as f:
                json.dump(dataset_class_names, f)

        # single / multi の両方に備えて, class ごとに複数枚持っておく
        images_per_class = args.use_real if args.use_real > 0 else max(8, args.real_images_per_step)

        collected_data_map = collect_target_images(
            dataset,
            target_ids_to_run,
            images_per_class,
            args.max_scan_limit,
            args.image_size,
            logger
        )
    else:
        target_ids_to_run = [int(t) for t in args.target_classes] if args.target_classes else []
        if args.dataset_type == "imagefolder":
            try:
                tmp_ds = ImageFolder(args.data_root)
                dataset_class_names = tmp_ds.classes
            except Exception as e:
                logger.log(f"Warning: Could not load ImageFolder for class names: {e}")
                dataset_class_names = []
        else:
            dataset_class_names = load_class_names_offline(args.cache_dir, args.dataset_type)

    if not dataset_class_names:
        default_count = 100 if "100" in args.dataset_type else 1000
        dataset_class_names = [str(i) for i in range(default_count)]

    models_list = []
    proj_dims = args.projection_dims
    if len(proj_dims) == 1:
        proj_dims = proj_dims * len(args.encoder_names)

    n_classes_adaptive = len(dataset_class_names)
    logger.log(f"Initializing Models with num_classes={n_classes_adaptive} (Dataset: {args.dataset_type})...")

    for name, p_dim in zip(args.encoder_names, proj_dims):
        m = EncoderClassifier(
            encoder_model=name,
            freeze_encoder=True,
            num_classes=n_classes_adaptive,
            feature_source=args.feature_source,
            projection_dim=p_dim,
            feature_layer=args.feature_layer,
        )
        m.to(device)
        m.eval()
        models_list.append(m)

    evaluator = Evaluator(device, use_ocr=(not args.disable_ocr), disable_all=args.no_evaluation)

    total_steps = len(target_ids_to_run) * len(args.experiments) * args.num_generations * args.num_iterations
    global_pbar = tqdm(total=total_steps, unit="step", desc="Total Progress")
    evaluation_results = []

    chunk_size = args.batch_opt_size
    for chunk_start in range(0, len(target_ids_to_run), chunk_size):
        batch_classes = target_ids_to_run[chunk_start:chunk_start + chunk_size]
        logger.log(f"--- Processing Batch: Classes {batch_classes} ---")

        batch_real_image_pools_cpu = []
        batch_ref_clean_tensors = []
        batch_class_names = []
        batch_safe_cls_names = []

        for target_cls in batch_classes:
            class_name = dataset_class_names[target_cls] if target_cls < len(dataset_class_names) else str(target_cls)
            safe_cls_name = sanitize_dirname(class_name)

            batch_class_names.append(class_name)
            batch_safe_cls_names.append(safe_cls_name)

            pil_images_list = collected_data_map.get(target_cls, [])

            if len(pil_images_list) == 0:
                logger.log(f"[Warning] No real images found for class {target_cls}.")
                batch_real_image_pools_cpu.append(torch.empty(0, 3, args.image_size, args.image_size))
                batch_ref_clean_tensors.append(None)
                continue

            ref_clean_tensor_cpu = to_tensor(
                pil_images_list[0].convert("RGB").resize((args.image_size, args.image_size))
            ).unsqueeze(0)

            real_pool_tensor_cpu = prepare_real_image_pool_tensor(
                pil_images_list=pil_images_list,
                image_size=args.image_size,
                overlay_text=args.overlay_text,
                font_scale=args.font_scale,
                text_color=args.text_color,
            )

            batch_real_image_pools_cpu.append(real_pool_tensor_cpu)
            batch_ref_clean_tensors.append(ref_clean_tensor_cpu)

        for exp_str in args.experiments:
            exp_name, weights_str = exp_str.split(":")
            weights = [float(w) for w in weights_str.split(",")]
            manage_model_allocation(models_list, weights, device)

            batch_save_dirs_base = []
            for b_idx, target_cls in enumerate(batch_classes):
                safe_cls_name = batch_safe_cls_names[b_idx]
                save_dir = os.path.join(args.output_dir, exp_name, f"{target_cls}_{safe_cls_name}")
                os.makedirs(save_dir, exist_ok=True)
                batch_save_dirs_base.append(save_dir)

            for gen_idx in range(args.num_generations):
                current_seed = args.seed + gen_idx
                set_seed(current_seed)

                batch_gen_subdirs = []
                for save_dir in batch_save_dirs_base:
                    gen_subdir = os.path.join(save_dir, f"gen_{gen_idx:02d}")
                    os.makedirs(gen_subdir, exist_ok=True)
                    batch_gen_subdirs.append(gen_subdir)

                lgm = MultiModelGM(models_list, weights, batch_classes, args, device)
                final_batch_img, best_batch_img, metrics = lgm.run(
                    real_image_pools_cpu=batch_real_image_pools_cpu,
                    save_dirs=batch_gen_subdirs,
                    class_names=batch_class_names,
                    logger=logger,
                    global_pbar=global_pbar,
                    gen_idx=gen_idx,
                )

                target_img_batch = best_batch_img if best_batch_img is not None else final_batch_img
                target_img_batch = target_img_batch.to(device)

                for b_idx, target_cls in enumerate(batch_classes):
                    target_img_tensor = target_img_batch[b_idx].unsqueeze(0)
                    gen_subdir = batch_gen_subdirs[b_idx]
                    save_dir = batch_save_dirs_base[b_idx]
                    class_name = batch_class_names[b_idx]

                    save_image(target_img_tensor, os.path.join(gen_subdir, "result.png"))
                    save_image(target_img_tensor, os.path.join(save_dir, f"result_gen{gen_idx:02d}.png"))

                    with open(os.path.join(gen_subdir, "metrics.json"), "w") as f:
                        json.dump(
                            {
                                "args": vars(args),
                                "metrics": metrics,
                            },
                            f,
                            indent=2
                        )

                    vis_score, _, _ = evaluator.calc_visual_score(target_img_tensor, target_cls)
                    text_score, _ = evaluator.calc_text_score(target_img_tensor, args.overlay_text)

                    ssim_val, lpips_val = 0.0, 0.0
                    ref_clean_tensor_cpu = batch_ref_clean_tensors[b_idx]
                    if ref_clean_tensor_cpu is not None:
                        ssim_val, lpips_val = evaluator.calc_similarity(target_img_tensor, ref_clean_tensor_cpu)

                    evaluation_results.append({
                        "exp_name": exp_name,
                        "class_id": target_cls,
                        "class_name": class_name,
                        "gen_idx": gen_idx,
                        "result_visual_score": vis_score,
                        "result_text_score": text_score,
                        "ssim": ssim_val,
                        "lpips": lpips_val,
                    })

                    append_completion_log(args.output_dir, target_cls, class_name, exp_name, gen_idx)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    global_pbar.close()

    if evaluation_results:
        df = pd.DataFrame(evaluation_results)
        csv_path = os.path.join(args.output_dir, "final_evaluation.csv")
        df.to_csv(csv_path, index=False)

    logger.close()
