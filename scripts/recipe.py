import argparse
from dataclasses import dataclass
from datetime import datetime
import gc
import io
from pathlib import Path
import re
from typing import Any, Dict, Literal, Optional, Self

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import AUROC
import tqdm

from landslide.data import LandslideDataset, get_images
from landslide.losses import AutoCriterion
from landslide.model.registry import load_model
from landslide.torch_utils import (
    dataloader,
    device_memory_clear,
    device_memory_used,
    init_seeds,
    intersect_dicts,
)
from landslide.trackers import Tracker


@dataclass
class TrainConfig:
    model: str
    project: str
    name: str
    dataset: str | Path
    weights: str | Path
    criterion: str
    image_sz: int
    mask_sz: int
    monitor: str
    patience: int
    conf: float
    device: str
    save_dir: str
    save_period: int
    tracker: str
    val: str
    batch_size: int
    mode: Literal['min', 'max']
    resume: bool = False
    pretrained: bool = False
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.0005
    seed: int = 0
    deterministic: bool = False
    normalize: bool = False
    workers: int = 8
    # Focal loss
    # alpha: float = 0.75
    # gamma: float = 2.0


class TrainContext:
    config: TrainConfig
    model: nn.Module | nn.parallel.DistributedDataParallel
    device: torch.device
    tracker: Tracker
    save_dir: str
    fitness: float
    train_loader: DataLoader
    valid_loader: DataLoader | None
    metrics: dict[str, Any] = {}
    start_iteration: int = 0
    best_iteration: int = 0
    current_iteration: int = None
    stop: bool = False

    def __init__(self, config: TrainConfig):
        self.device = device = torch.device(config.device)
        config.workers = 0 if device.type in {"cpu", "mps"} else config.workers
        if config.weights is not None:
            config.weights = Path(config.weights)
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.fitness = float("-inf") if config.mode == "max" else float("inf")
        init_seeds(config.seed, deterministic=config.deterministic)

    @property
    def weights_dir(self) -> Path:
        name = self.config.name if self.config.name is not None else self.config.model
        w = self.save_dir / name / 'weights'
        w.mkdir(parents=True, exist_ok=True)
        return w
    
    @property
    def plt_dir(self) -> Path:
        name = self.config.name if self.config.name is not None else self.config.model
        p = self.save_dir / name / 'plots'
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    @property
    def last_checkpoint(self) -> Path:
        return self.weights_dir / "last.pth"
    
    @property
    def best_checkpoint(self) -> Path:
        return self.weights_dir / 'best.pth'
    
    @property
    def current_checkpoint(self) -> Path:
        return self.weights_dir / f"epoch_{self.current_iteration}.pt"
    
    def __iter__(self) -> Self:
        return self

    def __next__(self) -> int:
        if self.current_iteration is None:
            self.current_iteration = self.start_iteration
            return self.current_iteration
        elif self.current_iteration >= self.config.epochs - 1:
            raise StopIteration
        
        self.current_iteration += 1
        return self.current_iteration


def schedule_dataloaders(ctx: TrainContext):
    config = ctx.config
    train_dataset = LandslideDataset(config.dataset, config.image_sz, config.mask_sz, config.normalize, split="train")
    valid_dataset = LandslideDataset(config.dataset, config.image_sz, config.mask_sz, config.normalize, split=config.val)
    ctx.train_loader = dataloader(train_dataset, batch_size=config.batch_size, workers=config.workers, shuffle=True, mode='train')
    ctx.valid_loader = dataloader(valid_dataset, batch_size=config.batch_size, workers=config.workers, shuffle=False, mode='valid')
    print(f"Training with {len(train_dataset)} train and {len(valid_dataset)} validation samples with imgsz {config.image_sz}")
    return ctx


def schedule_resume_model(ctx: TrainContext):
    # TODO: check after load_model refactor
    weights = ctx.config.weights
    if weights is None or not weights.exists():
        return ctx

    ctx.config.pretrained = True # For logging porposes
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    msd = checkpoint["model"] if "model" in checkpoint else checkpoint
    csd = intersect_dicts(msd, ctx.model.state_dict())  # intersect
    ctx.model.load_state_dict(csd, strict=False)  # load
    print(f"Transferred {len(csd)}/{len(ctx.model.state_dict())} items from pretrained weights")
    if ctx.config.resume and ctx.config.pretrained:
        ctx.optimizer.load_state_dict(checkpoint["optimizer"])
        ctx.start_iteration = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {ctx.start_iteration}")
    return ctx


def schedule_setup_logging(ctx: TrainContext):
    ctx.tracker = Tracker.load(name=ctx.config.tracker, config=vars(ctx.config))
    if ctx.config.tracker == "wandb":
        ctx.tracker.run.define_metric(ctx.config.monitor, summary=ctx.config.mode)
        ctx.tracker.run.define_metric("train/loss", summary="min")
        ctx.tracker.run.define_metric("valid/F1 (pixel)", summary="max")
        ctx.tracker.run.define_metric("valid/F1 (patch)", summary="max")    
        import wandb
        ctx.wb_table = wandb.Table(columns=["Image", "Epoch"])    
    return ctx


def schedule_train_epoch(ctx: TrainContext):
    criterion = ctx.criterion
    device = ctx.device
    loader = ctx.train_loader
    optimizer = ctx.optimizer
    model = ctx.model
    epoch = ctx.current_iteration

    running_loss = 0.0
    N = len(criterion)
    # DIRTY HACK: shorten the name of the criterion
    # trunacate the names longer than 11 characters to 8 + "..."
    names = [name[:5] + "..." if len(name) > 11 else name for name in criterion.names]
    key = "weighted_binary_cross_entropy"
    if key in criterion.names:
        names[criterion.names.index(key)] = "WBCE"
    key = "binary_cross_entropy"
    if key in criterion.names:
        names[criterion.names.index(key)] = "BCE"

    print(("\n" + "%11s" * (5 + N)) % ("Epoch", "GPU_mem", "Loss", *names, "Instances", "Size"))
    mean_losses = torch.zeros(N, device=device)
    progress = tqdm.tqdm(enumerate(loader), total=len(loader))
    ctx.model.train()
    optimizer.zero_grad()
    for i, batch in progress:
        targets = batch["target"].to(device, non_blocking=True, dtype=torch.float32)
        inputs = batch["input"].to(device, non_blocking=True).float()
        # Forward
        preds = model(inputs)  # (B, C, H, W) where C = number of classes
        preds = F.interpolate(preds, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        aggr_loss, losses = criterion(preds, targets)
        aggr_loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        running_loss = (running_loss * i + aggr_loss.item()) / (i + 1)
        mean_losses = (mean_losses * i + losses) / (i + 1)
        mem = f"{device_memory_used(device):.3g}G"

        progress.set_description(
            ("%11s" * 2 + "%11.4g" * (N + 3))
            % (
                f"{epoch + 1}/{ctx.config.epochs}",
                mem,
                running_loss,
                *mean_losses.tolist(),
                targets.shape[0],
                inputs.shape[-1],
            )
        )

    ctx.metrics = {f"train/{name}": cl.item() for name, cl in zip(criterion.names, mean_losses)}
    ctx.metrics['train/norm'] = norm
    ctx.metrics["train/loss"] = running_loss
    return ctx


@torch.inference_mode()
def schedule_valid_epoch(ctx: TrainContext):
    device = ctx.device
    loader = ctx.valid_loader
    model = ctx.model

    assert loader is not None, "Validation loader is None"
    running_loss = 0.0
    pixel_TP = pixel_FP = pixel_TN = pixel_FN = 0
    patch_TP = patch_FP = patch_TN = patch_FN = 0
    auroc_metric = AUROC(task="binary")

    mean = torch.tensor(loader.dataset.mean).view(3, 1, 1)
    std = torch.tensor(loader.dataset.std).view(3, 1, 1)

    normalize = loader.dataset.normalize
    loader.dataset.normalize = False
    
    print(("\n" + "%11s" * 4) % ("Precision", "Recall", "Accuracy", "F1"))
    progress = tqdm.tqdm(enumerate(loader), total=len(loader))
    ctx.model.eval()
    for i, batch in progress:
        from torchvision.transforms.v2 import functional as V
        
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True, dtype=torch.float32)
        logits = model(V.normalize(inputs, mean=mean, std=std) if normalize else inputs)  # (B, CLS, H, W)
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        running_loss = (running_loss * i + loss.item()) / (i + 1)
        probs = torch.sigmoid(logits)

        # AUC-ROC
        targets = targets.long()
        auroc_metric.update(probs.flatten(), targets.flatten())

        # Classification metrics
        preds = (probs > 0.5).long()
        # --- Pixel-wise metrics ---
        pixel_TP += ((preds == 1) & (targets == 1)).sum().item()
        pixel_FP += ((preds == 1) & (targets == 0)).sum().item()
        pixel_TN += ((preds == 0) & (targets == 0)).sum().item()
        pixel_FN += ((preds == 0) & (targets == 1)).sum().item()

        # --- Patch-wise metrics ---
        # A patch is positive if it has at least one positive pixel
        pred_patch = (preds.view(preds.size(0), -1).sum(dim=1) > 0)
        target_patch = (targets.view(targets.size(0), -1).sum(dim=1) > 0)

        patch_TP += ((pred_patch == 1) & (target_patch == 1)).sum().item()
        patch_FP += ((pred_patch == 1) & (target_patch == 0)).sum().item()
        patch_TN += ((pred_patch == 0) & (target_patch == 0)).sum().item()
        patch_FN += ((pred_patch == 0) & (target_patch == 1)).sum().item()

        eps = 1e-8
        precision = pixel_TP / (pixel_TP + pixel_FP + eps)
        recall = pixel_TP / (pixel_TP + pixel_FN + eps)
        accuracy = (pixel_TP + pixel_TN) / (pixel_TP + pixel_TN + pixel_FP + pixel_FN + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        patch_precision = patch_TP / (patch_TP + patch_FP + eps)
        patch_recall = patch_TP / (patch_TP + patch_FN + eps)
        patch_accuracy = (patch_TP + patch_TN) / (patch_TP + patch_TN + patch_FP + patch_FN + eps)
        patch_f1 = 2 * patch_precision * patch_recall / (patch_precision + patch_recall + eps)
        plot_batch(ctx, batch, preds.cpu())
        # update description with conf matrix
        progress.set_description(("%11.4g" * 4) % (precision, recall, accuracy, f1))

    metrics = {
        "valid/loss": running_loss,
        "valid/Precision (pixel)": precision,
        "valid/Recall (pixel)": recall,
        "valid/Accuracy (pixel)": accuracy,
        "valid/F1 (pixel)": f1,
        "valid/AUC-ROC": auroc_metric.compute().item(),
        "valid/Precision (patch)": patch_precision,
        "valid/Recall (patch)": patch_recall,
        "valid/Accuracy (patch)": patch_accuracy,
        "valid/F1 (patch)": patch_f1,
    }

    pixel_total = pixel_TP + pixel_TN + pixel_FP + pixel_FN
    metrics['valid/TP'] = pixel_TP / pixel_total
    metrics['valid/TN'] = pixel_TN / pixel_total
    metrics['valid/FP'] = pixel_FP / pixel_total
    metrics['valid/FN'] = pixel_FN / pixel_total

    patch_total = patch_TP + patch_TN + patch_FP + patch_FN
    metrics['valid/TP (patch)'] = patch_TP / patch_total
    metrics['valid/TN (patch)'] = patch_TN / patch_total
    metrics['valid/FP (patch)'] = patch_FP / patch_total
    metrics['valid/FN (patch)'] = patch_FN / patch_total
    ctx.metrics.update(metrics)
    return ctx


def metadata(path: Path) -> Optional[Dict[str, int]]:
    match = re.search(r"_(\d+)_(\d+)\..*$", path.name)
    if match:
        return {
            'path': path,
            'left': int(match.group(1)),
            'top': int(match.group(2))
        }
    else:
        raise ValueError(f"Filename {path.name} does not match expected pattern for patch coordinates.")


def merge_patches(path: Path, patch_size: int = 512) -> Image.Image:
    """
    Merge patches back to original image.

    Args:
        path: Directory containing patches
        patch_size: Original patch size before resize (default: 512)
    """
    ms = map(Path, get_images(path))
    ms = list(map(metadata, ms))

    # Load patches and upscale them back to original patch size
    rs = []
    for patch_meta in ms:
        patch = Image.open(patch_meta['path'])
        patch = patch.resize((patch_size, patch_size), Image.Resampling.NEAREST)
        rs.append(patch)

    # Calculate canvas size based on max coordinates + patch size
    r_h, r_w = 0, 0
    for m in ms:
        left, top = m['left'], m['top']
        r_w = max(r_w, left + patch_size)
        r_h = max(r_h, top + patch_size)

    # Create canvas - use 'L' mode for grayscale masks
    mode = rs[0].mode if rs else "RGB"
    r_img = Image.new(mode, (r_w, r_h), (0, 0, 0) if mode == "RGB" else 0)

    # Paste patches at their original coordinates
    for m, r in zip(ms, rs):
        left, top = m['left'], m['top']
        r_img.paste(r, (left, top))

    return r_img



def plot_batch(ctx: TrainContext, batch: dict, preds: torch.Tensor):
    # Tmp patches dirs
    p_dst = ctx.plt_dir / f"epoch_{ctx.current_iteration}" / "patches"
    i_dst = ctx.plt_dir / 'image' / 'patches'
    t_dst = ctx.plt_dir / 'mask' / 'patches'

    # Reconstructed images
    p_img = p_dst.parent / 'pred.png'
    i_img = i_dst.parent / 'image.png'
    t_img = t_dst.parent / 'mask.png'

    data = {
        'img': {'dst': ctx.plt_dir / f"epoch_{ctx.current_iteration}" / "patches", 'name': 'image'},
        'mask': {'dst': ctx.plt_dir / 'image' / 'patches', 'name': 'mask'},
        'pred': {'dst': ctx.plt_dir / 'mask' / 'patches', 'name': 'pred'},
    }

    for k, v in data.items():
        v['dst'].mkdir(exist_ok=True, parents=True)
        data[k]['image'] = v['dst'].parent / f"{v['name']}.png"

    for k, v in data.items():
        if not v['image'].exists():
            reconstruced = merge_patches(v['dst'])
            reconstruced.save(data[k]['image'])
            data[k]['image'] = reconstruced
            v['dst'].unlink()


    p_dst.mkdir(exist_ok=True, parents=True)
    i_dst.mkdir(exist_ok=True, parents=True)
    t_dst.mkdir(exist_ok=True, parents=True)
    
    for sample_idx in range(batch["input"].size(0)):
        name = batch["image_path"][sample_idx].name
        if not i_img.exists():
            si = batch["input"][sample_idx] * 255
            si = si.numpy().transpose(1, 2, 0).astype(np.uint8)
            Image.fromarray(si).save(i_dst / name)

        if not t_img.exists():
            st = batch["target"][sample_idx].squeeze().numpy().astype(np.uint8) * 255
            Image.fromarray(st).save(t_dst / name)

        if not p_img.exists():
            so = preds[sample_idx].squeeze().numpy().astype(np.uint8) * 255
            Image.fromarray(so).save(p_dst / name)

    for dst, img in zip([p_dst, i_dst, t_dst], [p_img, i_img, t_img]):
        if not img.exists():
            # Reconstruct from patches
            merge_patches(dst).save(img)
            # Clean up to free some space
            dst.unlink()


    if ctx.config.tracker == "wandb":
        import wandb
        ctx.wb_table.add_data(wandb.Image(
            Image.open(i_img).convert('RGB'),
            masks={
                "ground_truth": {
                    "mask_data": np.array(Image.open(t_img).convert('L')) / 255.0,
                    "class_labels": {0: "background", 1: "landslide"}
                },
                "prediction": {
                    "mask_data": np.array(Image.open(p_img).convert('L')) / 255.0,
                    "class_labels": {0: "background", 1: "landslide"}
                }
            }
        ), ctx.current_iteration)




def schedule_early_stopping(ctx: TrainContext):
    x = ctx.metrics[ctx.config.monitor]
    y = ctx.fitness
    if (ctx.config.mode == "max" and x >= y) or (ctx.config.mode == "min" and x <= y):
        ctx.fitness = x
        ctx.best_iteration = ctx.current_iteration
    if ctx.current_iteration - ctx.best_iteration == ctx.config.patience:
        print(f"Triggered Early Stopping at epoch {ctx.current_iteration + 1}")
        ctx.stop = True
    return ctx


def schedule_model_checkpointing(ctx: TrainContext):
    buffer = io.BytesIO()
    torch.save({
        "epoch": ctx.current_iteration,
        "model": ctx.model.state_dict(),
        "optimizer": ctx.optimizer.state_dict(),
        "metrics": ctx.metrics,
        "config": vars(ctx.config),
        "date": datetime.now().isoformat(),
    }, buffer)
    ckpt = buffer.getvalue()
    # ctx.last_checkpoint.write_bytes(ckpt)
    if ctx.current_iteration == ctx.best_iteration:
        ctx.best_checkpoint.write_bytes(ckpt)
        ctx.tracker.log_model(ctx.best_checkpoint, aliases=["best"])
    if ctx.config.save_period > 0 and (ctx.current_iteration + 1) % ctx.config.save_period == 0:
        ctx.current_checkpoint.write_bytes(ckpt)

    return ctx


def schedule_train(config: TrainConfig):
    ctx = TrainContext(config)
    ctx = schedule_dataloaders(ctx)
    ctx.model = load_model(ctx.model)
    ctx.criterion = AutoCriterion(ctx.config.criterion, {"nc": 1, "pos_weight": ctx.train_loader.dataset.data['patch_weight']})
    ctx.optimizer = torch.optim.AdamW(ctx.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    ctx = schedule_resume_model(ctx)
    ctx = schedule_setup_logging(ctx)
    ctx.model = ctx.model.to(ctx.device)

    for _ in ctx:
        ctx = schedule_train_epoch(ctx)
        ctx = schedule_valid_epoch(ctx)
        ctx.tracker.log(ctx.metrics, step=ctx.current_iteration)
        ctx = schedule_early_stopping(ctx)
        ctx = schedule_model_checkpointing(ctx)
        gc.collect()
        device_memory_clear(ctx.device)
        if ctx.stop:
            break

    # Log wandb table at the end of training
    if ctx.config.tracker == "wandb":
        ctx.tracker.log({"valid/predictions": ctx.wb_table})

    # schedule_final_validation(ctx)


if __name__ == "__main__":
    import argparse

    from landslide.utils import yaml_load
    parser = argparse.ArgumentParser(description="Train a binary semantic segmentation model on landslide imagery.")
    parser.add_argument("--config", type=str)
    # Model and dataset
    parser.add_argument("--model", type=str, choices=["unet", "segformer"], help="Name of the model architecture to use (e.g., 'unet', 'fcn').")
    parser.add_argument("--project", type=str, help="Project name for tracking and logging.")
    parser.add_argument("--name", type=str, help="Name of the training run.")
    parser.add_argument("--dataset", type=str, help="Identifier or path for the dataset index.")
    parser.add_argument("--weights", type=str, help="Path to pretrained model weights to initialize training.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, help="Whether to resume training from existing weights checkpoint.")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, help="Total number of training epochs.")
    parser.add_argument("--batch_size", type=int, help="Batch size for training and validation.")
    parser.add_argument("--lr", type=float, help="Initial learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, help="Weight decay (L2 regularization) factor.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.") # Experiments were done setting seed to 1337
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, help="Enable deterministic behavior for reproducible results.")
    parser.add_argument("--criterion", type=str, help="Loss criterion name (e.g., 'binary_cross_entropy', 'weighted_binary_cross_entropy', etc.).")

    # Data settings
    parser.add_argument("--image_sz", type=int, help="Input image spatial size (height and width).")
    parser.add_argument("--mask_sz", type=int, help="Output mask spatial size.")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, help="Normalize input images using dataset mean and std.")

    # Monitoring and early stopping
    parser.add_argument("--monitor", type=str, help="Metric name to monitor for model checkpointing and early stopping.")
    parser.add_argument("--patience", type=int, help="Number of epochs with no improvement before early stopping.")
    parser.add_argument("--mode", choices=["max", "min"], help="Mode for monitoring metric ('max' to maximize, 'min' to minimize).")

    # Miscellaneous
    parser.add_argument("--conf", type=float, help="Confidence threshold for converting logits to binary predictions.")
    parser.add_argument("--workers", type=int, help="Number of dataloader worker processes.")
    parser.add_argument("--device", type=str, help="Device identifier (e.g., 'cuda:0', 'cpu', 'mps:0').")
    parser.add_argument("--save_dir", type=str, help="Directory where training runs and checkpoints will be saved.")
    parser.add_argument("--save_period", type=int, help="Epoch interval for periodic checkpoint saving (-1 to disable).")
    parser.add_argument("--tracker", type=str, help="Tracker type for logging (e.g., 'wandb').")
    parser.add_argument("--val", type=str, help="Fold key to choose data for validation.")
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    if args["config"]:
        base = yaml_load(args.pop("config"))['params']
        args = {**base, **args}  # Command-line args override config file

    schedule_train(TrainConfig(**args))
