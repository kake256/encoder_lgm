import os
import sys
import argparse
import copy
import json
import re
import glob
import random # 【追加】randomモジュール
import numpy as np # 【追加】numpyモジュール
from PIL import Image
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torchvision.transforms as T
from torchvision.utils import save_image
from torchvision.transforms.functional import resize, to_tensor
from tqdm import tqdm
from datasets import load_dataset
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from multi_model_utils import EncoderClassifier, PyramidGenerator, TVLoss

# ==============================================================================
# Helper Functions
# ==============================================================================

# 【追加】乱数シード固定関数
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 再現性を重視する場合は以下も有効にするが、速度が落ちる可能性あり
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sanitize_dirname(name):
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    return safe_name.strip('_')

def scan_rta100_classes(data_root, mode):
    if not data_root: return {}, []
    
    files = glob.glob(os.path.join(data_root, "*.jpg")) + \
            glob.glob(os.path.join(data_root, "*.jpeg")) + \
            glob.glob(os.path.join(data_root, "*.png"))
            
    print(f"Scanning RTA100... Found {len(files)} files.")
    
    class_map = {}
    pattern = re.compile(r'label=(.+?)_text=(.+)')
    
    for f in files:
        fname = os.path.basename(f)
        base = os.path.splitext(fname)[0]
        
        match = pattern.search(base)
        if match:
            label_val = match.group(1).lower().strip()
            text_val = match.group(2).lower().strip()
            target = label_val if mode == 'label' else text_val
            
            if target:
                if target not in class_map: class_map[target] = []
                class_map[target].append(f)
            
    sorted_classes = sorted(list(class_map.keys()))
    
    if len(sorted_classes) > 0:
        print(f"DEBUG: First 10 recognized classes: {sorted_classes[:10]}")
    else:
        print("DEBUG: No classes recognized! Check filenames.")
        
    return class_map, sorted_classes

# ==============================================================================
# Logger Class
# ==============================================================================
class Logger:
    def __init__(self, log_path=None):
        self.log_path = log_path
        if log_path:
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self.file = open(log_path, 'w', encoding='utf-8')
        else:
            self.file = None

    def log(self, message):
        tqdm.write(message)
        if self.file:
            self.file.write(message + '\n')
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()

# ==============================================================================
# MultiModelGM クラス
# ==============================================================================

class MultiModelGM:
    def __init__(self, models, model_weights, target_class, args, device, initial_image=None):
        self.models = models
        self.model_weights = model_weights
        self.target_class = target_class
        self.args = args
        self.device = device
        
        self.generator = PyramidGenerator(
            target_size=args.image_size,
            start_size=args.pyramid_start_res,
            activation='sigmoid',
            initial_image=initial_image,
            noise_level=args.seed_noise_level
        ).to(device)
        
        self.optimizer = self._init_optimizer()
        
        self.augmentor = T.Compose([
            T.RandomResizedCrop(args.image_size, scale=(0.8, 1.0), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            T.ColorJitter(0.1, 0.1, 0.1, 0.05),
        ])
        
        self.scaler = torch.cuda.amp.GradScaler()
        self.tv_loss_fn = TVLoss().to(device)

    def _init_optimizer(self):
        return optim.Adam(self.generator.parameters(), lr=self.args.lr)

    def preprocess(self, img):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        return (img - mean) / std

    def get_grads(self, model, inputs, create_graph=False):
        params = list(model.classifier.parameters())
        logits = model(inputs)
        targets = torch.tensor([self.target_class] * inputs.size(0), device=self.device)
        loss = F.cross_entropy(logits, targets)
        grads = torch.autograd.grad(loss, params, create_graph=create_graph)
        flat_grad = torch.cat([g.view(-1) for g in grads])
        return flat_grad

    def optimize_step(self, real_images_pool):
        self.optimizer.zero_grad()
        syn_image = self.generator()
        
        with torch.no_grad():
            indices = torch.randperm(len(real_images_pool))[:self.args.num_ref_images]
            real_batch = real_images_pool[indices].detach()
            aug_real = self.augmentor(real_batch)
            inp_real = self.preprocess(aug_real)

        syn_batch_list = []
        for _ in range(self.args.augs_per_step):
            syn_batch_list.append(self.augmentor(syn_image))
        syn_batch = torch.cat(syn_batch_list, dim=0)
        inp_syn = self.preprocess(syn_batch)

        total_grad_loss = 0.0
        
        per_model_sims = {}
        
        for i, model in enumerate(self.models):
            weight = self.model_weights[i]
            if weight == 0: continue

            model.reset_classifier()
            
            with autocast():
                target_grad = self.get_grads(model, inp_real, create_graph=False)
            
            with autocast():
                syn_grad = self.get_grads(model, inp_syn, create_graph=True)
            
            sim = F.cosine_similarity(target_grad.unsqueeze(0).detach(), syn_grad.unsqueeze(0)).mean()
            loss_k = 1.0 - sim
            total_grad_loss += loss_k * weight
            
            model_name = self.args.encoder_names[i] if i < len(self.args.encoder_names) else f"model_{i}"
            per_model_sims[model_name] = sim.item()

        loss_tv = self.tv_loss_fn(syn_image)
        total_loss = (total_grad_loss * self.args.weight_grad) + (loss_tv * self.args.weight_tv)
        
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # 【変更】total_loss.item() も返す (ベストショット判定用)
        return total_grad_loss.item(), loss_tv.item(), per_model_sims, total_loss.item()

    def run(self, real_images_pool, save_dir, class_names, logger, global_pbar):
        logger.log(f"[{self.target_class}] Optimization Start. Pool size: {len(real_images_pool)}")
        loss_history = []
        
        # 【追加】ベストショット保存用変数
        best_loss = float('inf')
        best_img_tensor = None

        local_pbar = tqdm(range(self.args.num_iterations), desc=f"Current Exp", leave=False, position=1, dynamic_ncols=True)
        
        for i in local_pbar:
            if i > 0 and i % self.args.pyramid_grow_interval == 0:
                if self.generator.extend():
                    self.optimizer = self._init_optimizer()
            
            # 【変更】total_loss も受け取る
            l_grad, l_tv, model_sims, current_total_loss = self.optimize_step(real_images_pool)
            
            # 【追加】ベストロスの更新と画像の保持
            if current_total_loss < best_loss:
                best_loss = current_total_loss
                # 現在の画像をCPUにコピーして保持 (detach)
                best_img_tensor = self.generator().detach().cpu()

            if i % 100 == 0:
                if logger.file:
                    logger.file.write(f"__PROGRESS__ {i}/{self.args.num_iterations} {l_grad:.4f}\n")

            step_metrics = {"loss_grad": l_grad, "loss_tv": l_tv, "total_loss": current_total_loss} # total_lossも記録
            for m_name, m_sim in model_sims.items():
                step_metrics[f"sim_{m_name}"] = m_sim
                
            loss_history.append(step_metrics)
            
            local_pbar.set_description(f"LossG: {l_grad:.3f} TV: {l_tv:.4f} Res: {self.generator.levels[-1].shape[-1]}")
            
            global_pbar.update(1)
            global_pbar.set_postfix(grad_loss=f"{l_grad:.4f}")

            if i % 500 == 0:
                with torch.no_grad():
                    save_image(self.generator().detach().cpu(), os.path.join(save_dir, f"step_{i:04d}.png"))

        # 最終画像を保存
        final_img = self.generator().detach().cpu()
        
        # 【追加】ベスト画像も返す
        return final_img, best_img_tensor, {"loss_history": loss_history}


# ==============================================================================
# Main
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Model LGM")
    parser.add_argument("--encoder_names", type=str, nargs='+', required=True, help="List of encoder models")
    parser.add_argument("--projection_dims", type=int, nargs='+', default=[2048], help="Projection dims corresponding to encoders")
    parser.add_argument("--experiments", type=str, nargs='+', required=True, help="List of experiments (name:w1,w2,w3)")
    parser.add_argument("--output_dir", type=str, default="./lgm_multi_results", help="Output directory")
    parser.add_argument("--target_classes", type=str, nargs='+', default=[], help="Target class indices or names")
    parser.add_argument("--dataset_type", type=str, default="imagenet", choices=["imagenet", "rta100"], help="Dataset type")
    parser.add_argument("--data_root", type=str, default=None, help="Root dir for RTA100")
    parser.add_argument("--rta_mode", type=str, default="label", choices=["label", "text"], help="Parse mode for RTA100")
    parser.add_argument("--initial_image_path", type=str, default=None)
    parser.add_argument("--seed_noise_level", type=float, default=0.0)
    parser.add_argument("--num_ref_images", type=int, default=10, help="Real images batch size per iter")
    parser.add_argument("--augs_per_step", type=int, default=32)
    parser.add_argument("--num_iterations", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--pyramid_start_res", type=int, default=16)
    parser.add_argument("--pyramid_grow_interval", type=int, default=400)
    parser.add_argument("--weight_grad", type=float, default=1.0)
    parser.add_argument("--weight_tv", type=float, default=0.00025)
    
    # 【追加】乱数シード設定引数
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    parser.add_argument("--log_file", type=str, default="training_log.txt", help="Path to save the log file")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 【追加】シード固定
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = Logger(args.log_file)
    
    if torch.cuda.is_available():
        logger.log(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True # set_seedでFalseにした場合はここで上書きしないよう注意。速度優先ならTrueのままでOK

    # -----------------------------------------------------------
    # データセット準備
    # -----------------------------------------------------------
    rta_class_map = {}
    rta_class_names = []
    
    if args.dataset_type == "rta100":
        if not args.data_root:
            raise ValueError("RTA100 requires --data_root path.")
        rta_class_map, rta_class_names = scan_rta100_classes(args.data_root, args.rta_mode)
        num_classes = len(rta_class_names)
        logger.log(f"[RTA100] Found {num_classes} unique classes (Mode: {args.rta_mode})")
    else:
        num_classes = 1000
        logger.log("[ImageNet] Setting num_classes = 1000")

    # -----------------------------------------------------------
    # モデルリストの初期化
    # -----------------------------------------------------------
    models_list = []
    proj_dims = args.projection_dims
    if len(proj_dims) == 1:
        proj_dims = proj_dims * len(args.encoder_names)
    
    logger.log("Initializing Models...")
    for name, p_dim in zip(args.encoder_names, proj_dims):
        logger.log(f"  - Loading {name} (Proj: {p_dim})...")
        m = EncoderClassifier(
            encoder_model=name,
            freeze_encoder=True,
            num_classes=num_classes,
            projection_dim=p_dim
        )
        m.to(device)
        m.eval()
        models_list.append(m)

    # -----------------------------------------------------------
    # ターゲットID決定
    # -----------------------------------------------------------
    target_ids_to_run = []
    dataset_class_names = []

    if args.dataset_type == "rta100":
        name_to_id = {name: i for i, name in enumerate(rta_class_names)}
        dataset_class_names = rta_class_names
        
        if not args.target_classes:
            target_ids_to_run = list(range(num_classes))
        else:
            for t_str in args.target_classes:
                t_lower = t_str.lower().strip()
                if t_lower in name_to_id:
                    target_ids_to_run.append(name_to_id[t_lower])
                else:
                    found = False
                    for candidate in name_to_id.keys():
                        if t_lower == candidate or (t_lower in candidate):
                            target_ids_to_run.append(name_to_id[candidate])
                            logger.log(f"Info: '{t_str}' match found as '{candidate}'")
                            found = True
                            break
                    if not found:
                        logger.log(f"Warning: Class '{t_str}' not found in RTA100.")
    else:
        logger.log("Loading ImageNet validation set...")
        try:
            dataset = load_dataset("imagenet-1k", split="validation", streaming=False)
            dataset_class_names = dataset.features['label'].names
        except:
            logger.log("Warning: Failed to load HF dataset. Using dummy data structure.")
            dataset = []
            dataset_class_names = [f"class_{i}" for i in range(1000)]

        for t in args.target_classes:
            try:
                target_ids_to_run.append(int(t))
            except:
                logger.log(f"Skipping invalid target ID: {t}")

    if not target_ids_to_run:
        logger.log("No valid target classes found. Exiting.")
        logger.close()
        exit()

    initial_image_tensor = None
    if args.initial_image_path and os.path.exists(args.initial_image_path):
        pil_img = Image.open(args.initial_image_path).convert('RGB')
        initial_image_tensor = to_tensor(resize(pil_img, (args.image_size, args.image_size))).unsqueeze(0).to(device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------
    # 統合プログレスバー設定
    # -----------------------------------------------------------
    total_steps = len(target_ids_to_run) * len(args.experiments) * args.num_iterations
    logger.log(f"Starting experiments for {len(target_ids_to_run)} classes...")
    logger.log(f"Total Steps to run: {total_steps}")
    
    global_pbar = tqdm(total=total_steps, unit="step", desc="Total Progress", position=0, dynamic_ncols=True)

    # -----------------------------------------------------------
    # 実行ループ
    # -----------------------------------------------------------
    for target_cls in target_ids_to_run:
        real_images_pool = []
        MAX_POOL_SIZE = 200
        
        # 画像収集
        if args.dataset_type == "rta100":
            cls_name = rta_class_names[target_cls]
            file_list = rta_class_map[cls_name]
            use_files = file_list[:MAX_POOL_SIZE]
            for fpath in use_files:
                try:
                    img = Image.open(fpath).convert('RGB')
                    img = resize(img, [args.image_size, args.image_size])
                    real_images_pool.append(to_tensor(img))
                except:
                    continue
        else:
            for item in dataset:
                if item['label'] == target_cls:
                    img = item['image'].convert('RGB')
                    img = resize(img, [args.image_size, args.image_size])
                    real_images_pool.append(to_tensor(img))
                    if len(real_images_pool) >= MAX_POOL_SIZE: break
        
        if len(real_images_pool) == 0:
            logger.log(f"Skipping class {target_cls} (No images)")
            steps_skipped = len(args.experiments) * args.num_iterations
            global_pbar.update(steps_skipped)
            continue
            
        real_images_pool = torch.stack(real_images_pool).to(device)
        class_name = dataset_class_names[target_cls]

        # 実験ループ
        for exp_str in args.experiments:
            exp_name, weights_str = exp_str.split(':')
            weights = [float(w) for w in weights_str.split(',')]
            
            save_name = f"{target_cls}_{sanitize_dirname(class_name)}"
            exp_save_dir = os.path.join(args.output_dir, exp_name, save_name)
            os.makedirs(exp_save_dir, exist_ok=True)

            logger.log(f"\n>>> Class: {class_name} | Exp: {exp_name} | Weights: {weights}")

            if not os.path.exists(os.path.join(exp_save_dir, "ref_pool.png")):
                save_image(real_images_pool[:20], os.path.join(exp_save_dir, "ref_pool.png"), nrow=5)

            # 最適化実行
            lgm = MultiModelGM(models_list, weights, target_cls, args, device, initial_image=initial_image_tensor)
            
            # 【変更】戻り値が増えたので受け取り方を修正
            final_img, best_img, metrics = lgm.run(real_images_pool, exp_save_dir, dataset_class_names, logger, global_pbar)
            
            # 最終画像とベスト画像を保存
            save_image(final_img, os.path.join(exp_save_dir, "final_multi_model.png"))
            if best_img is not None:
                save_image(best_img, os.path.join(exp_save_dir, "best_multi_model.png"))
            
            with open(os.path.join(exp_save_dir, "metrics.json"), 'w') as f:
                json.dump({"args": vars(args), "metrics": metrics}, f, indent=2)

    global_pbar.close()
    logger.log("All Experiments Completed.")
    logger.close()
    
    sys.exit(0)