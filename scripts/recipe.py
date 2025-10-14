import argparse
import gc
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from landslide.data import LandslideDataset
from landslide.losses import AutoCriterion
from landslide.metrics import BinaryConfusionMatrix
from landslide.torch_utils import (dataloader, device_memory_clear,
                                   device_memory_used, init_seeds,
                                   intersect_dicts)
from landslide.trackers import Tracker


@dataclass
class TrainConfig:
    model: str
    project: str
    name: str
    dataset_content: str
    dataset_index: str
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
    current_iteration: int = 0
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
    def weights_dir(self):
        m = self.config.name if self.config.name is not None else self.config.model
        w = self.save_dir / m / 'weights'
        w.mkdir(parents=True, exist_ok=True)
        return w
    
    @property
    def last_checkpoint(self) -> Path:
        return self.weights_dir / "last.pth"
    
    @property
    def best_checkpoint(self) -> Path:
        return self.weights_dir / 'best.pth'
    
    @property
    def current_checkpoint(self) -> Path:
        return self.weights_dir / f"epoch_{self.current_iteration}.pt"


def schedule_dataloaders(ctx: TrainContext):
    config = ctx.config
    train_dataset = LandslideDataset(config.dataset_index, config.dataset_content, config.image_sz, config.mask_sz, config.normalize, split="train")
    valid_dataset = LandslideDataset(config.dataset_index, config.dataset_content, config.image_sz, config.mask_sz, config.normalize, split=config.val)
    if ctx.config.normalize:
        train_dataset.compute_stats()
        valid_dataset.mean = train_dataset.mean
        valid_dataset.std = train_dataset.std
    ctx.train_loader = dataloader(train_dataset, batch_size=config.batch_size, workers=config.workers, shuffle=True, mode='train')
    ctx.valid_loader = dataloader(valid_dataset, batch_size=config.batch_size, workers=config.workers, shuffle=False, mode='valid')
    print(f"Training with {len(train_dataset)} train and {len(valid_dataset)} validation samples with imgsz {config.image_sz}")
    return ctx


def schedule_load_model(ctx: TrainContext):
    assert ctx.train_loader is not None
    from landslide.model.segformer import (SegformerConfig,
                                           SegformerForSemanticSegmentation)
    from landslide.model.unet import UNet

    if ctx.config.model == "unet":
        ctx.model = UNet(nc=1, ch=3)
    elif ctx.config.model == "segformer":
        model_config = SegformerConfig()
        model_config.num_labels = 1
        model_config.num_channels = 3
        ctx.model = SegformerForSemanticSegmentation(model_config)
    return ctx


def schedule_resume_model(ctx: TrainContext):
    weights = ctx.config.weights
    if weights is None or not weights.exists():
        return ctx

    ctx.config.pretrained = True # For logging porposes
    if weights.name.endswith(".safetensors"):
        from safetensors.torch import load_file
        checkpoint = load_file(weights, device="cpu")
    else:
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
        ctx.tracker.run.define_metric(ctx.config.monitor, summary=config.mode)
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
    optimizer.zero_grad()
    for i, batch in progress:
        targets = batch["target"].to(device, non_blocking=True, dtype=torch.float32)
        inputs = batch["input"].to(device, non_blocking=True).float()
        # Forward
        preds = model(inputs)  # (B, C, H, W) where C = number of classes
        preds = F.interpolate(preds, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        aggr_loss, losses = criterion(preds, targets)
        aggr_loss.backward()
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
    ctx.metrics["train/loss"] = running_loss
    return ctx


def schedule_valid_epoch(ctx: TrainContext):
    criterion = ctx.criterion
    device = ctx.device
    loader = ctx.valid_loader
    model = ctx.model

    assert loader is not None, "Validation loader is None"
    running_loss = 0.0
    num_objectives = len(ctx.criterion)
    print(("\n" + "%11s" * 4) % ("Precision", "Recall", "Accuracy", "F1"))
    mean_losses = torch.zeros(num_objectives, device=device)

    confmat = BinaryConfusionMatrix()
    confmat.to(device)

    progress = tqdm.tqdm(enumerate(loader), total=len(loader))
    for i, batch in progress:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True, dtype=torch.float32)
        preds = model(inputs)  # (B, C, H, W) where C = number of classes
        preds = F.interpolate(preds, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        aggr_loss, losses = criterion(preds, targets)
        num_cls = preds.shape[1]
        mask = F.sigmoid(preds) > ctx.config.conf if num_cls == 1 else torch.argmax(preds, dim=1)
        mask = mask.to(torch.uint8)
        targets = targets.long()
        confmat(mask, targets)
        running_loss = (running_loss * i + aggr_loss.item()) / (i + 1)
        mean_losses = (mean_losses * i + losses.detach()) / (i + 1)

        # update description with conf matrix
        description = list(confmat.metrics().values())[-4:]
        progress.set_description(("%11.4g" * 4) % tuple(description))

    ctx.metrics["valid/loss"] = running_loss
    ctx.metrics.update(confmat.metrics(prefix="valid/"))
    for name, loss in zip(criterion.names, mean_losses):
        ctx.metrics[f"valid/{name}"] = loss.item()
    return ctx


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
    aliases = ["last"]
    ctx.last_checkpoint.write_bytes(ckpt)
    if ctx.current_iteration == ctx.best_iteration:
        ctx.best_checkpoint.write_bytes(ckpt)
        aliases.append("best")
    if ctx.config.save_period > 0 and (ctx.current_iteration + 1) % ctx.config.save_period == 0:
        ctx.current_checkpoint.write_bytes(ckpt)
    
    ctx.tracker.log_model(ckpt, aliases=aliases)
    return ctx


def schedule_clear_memory(ctx: TrainContext):
    gc.collect()
    device_memory_clear(ctx.device)
    return ctx


def schedule_train(config: TrainConfig):
    ctx = TrainContext(config)
    ctx = schedule_dataloaders(ctx)
    ctx = schedule_load_model(ctx)
    ctx.criterion = AutoCriterion(ctx.config.criterion, {})
    ctx.optimizer = torch.optim.AdamW(ctx.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    ctx = schedule_resume_model(ctx)
    ctx = schedule_setup_logging(ctx)
    ctx.model = ctx.model.to(ctx.device)

    for epoch in range(ctx.start_iteration, ctx.config.epochs):
        ctx.current_iteration = epoch
        ctx = schedule_train_epoch(ctx)
        ctx = schedule_valid_epoch(ctx)
        ctx = schedule_early_stopping(ctx)
        ctx = schedule_model_checkpointing(ctx)
        ctx = schedule_clear_memory(ctx)
        if ctx.stop:
            break
    # schedule_final_validation(ctx)
    

if __name__ == "__main__":
    import argparse

    from landslide.utils import yaml_load
    
    parser = argparse.ArgumentParser(description="Train a binary semantic segmentation model on landslide imagery.")
    parser.add_argument("--config", type=str)
    # Model and dataset
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "segformer"], help="Name of the model architecture to use (e.g., 'unet', 'fcn').")
    parser.add_argument("--project", type=str, default="landslide", help="Project name for tracking and logging.")
    parser.add_argument("--name", type=str, default=None, help="Name of the training run.")
    parser.add_argument("--dataset_content", type=str, help="Identifier or path for the dataset content.")
    parser.add_argument("--dataset_index", type=str, help="Identifier or path for the dataset index.")
    parser.add_argument("--weights", type=str, default=None, help="Path to pretrained model weights to initialize training.")
    parser.add_argument("--resume", action="store_true", help="Whether to resume training from existing weights checkpoint.")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Total number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay (L2 regularization) factor.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.") # Experiments were done setting seed to 1337
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic behavior for reproducible results.")
    parser.add_argument("--criterion", type=str, default="weighted_binary_cross_entropy", help="Loss criterion name (e.g., 'binary_cross_entropy', 'weighted_binary_cross_entropy', etc.).")

    # Data settings
    parser.add_argument("--image_sz", type=int, default=128, help="Input image spatial size (height and width).")
    parser.add_argument("--mask_sz", type=int, default=128, help="Output mask spatial size.")
    parser.add_argument("--normalize", action="store_true", help="Normalize input images using dataset mean and std.")

    # Monitoring and early stopping
    parser.add_argument("--monitor", type=str, default="valid/F1-Score", help="Metric name to monitor for model checkpointing and early stopping.")
    parser.add_argument("--patience", type=int, default=10, help="Number of epochs with no improvement before early stopping.")
    parser.add_argument("--mode", choices=["max", "min"], default="max", help="Mode for monitoring metric ('max' to maximize, 'min' to minimize).")

    # Miscellaneous
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for converting logits to binary predictions.")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader worker processes.")
    parser.add_argument("--device", type=str, default="cpu", help="Device identifier (e.g., 'cuda:0', 'cpu', 'mps:0').")
    parser.add_argument("--save_dir", type=str, default="./runs", help="Directory where training runs and checkpoints will be saved.")
    parser.add_argument("--save_period", type=int, default=-1, help="Epoch interval for periodic checkpoint saving (-1 to disable).")
    parser.add_argument("--tracker", type=str, default=None, help="Tracker type for logging (e.g., 'wandb').")
    parser.add_argument("--val", type=str, default="valid", help="Fold key to choose data for validation.")
    args = parser.parse_args()

    config = yaml_load(args.config)['params']
    config = TrainConfig(**config)
    schedule_train(config)
