import argparse
from datetime import datetime
import gc
import io
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from landslide.data import LandslideDataset, dataloader, dataset_read_config
from landslide.dtypes import IterableSimpleNamespace
from landslide.losses import AutoCriterion
from landslide.metrics import BinaryConfusionMatrix
from landslide.model import init_model
from landslide.torch_utils import device_memory_clear, device_memory_used, init_seeds
from landslide.trackers import _WANDB_AVAILABLE, Tracker, WandbTracker

# Configure logger
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)



def intersect_dicts(da, db, exclude=()):
    """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
    return {
        k: v
        for k, v in da.items()
        if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape
    }
    

def load_model(model: nn.Module, weights: Path, verbose: bool = True) -> nn.Module:
    from safetensors.torch import load_file
    checkpoint = torch.load(weights, map_location="cpu") if not weights.name.endswith(".safetensors") else load_file(weights, device="cpu")
    checkpoint = checkpoint["model"] if "model" in checkpoint else checkpoint
    csd = intersect_dicts(checkpoint, model.state_dict())  # intersect
    model.load_state_dict(csd, strict=False)  # load
    if verbose:
        print(f"Transferred {len(csd)}/{len(model.state_dict())} items from pretrained weights")
    return model


def train_epoch(model, hyp, loader, epoch, criterion: AutoCriterion, device, optimizer):
    running_loss = 0.0
    n_objectives = len(criterion)

    # DIRTY HACK: shorten the name of the criterion
    # trunacate the names longer than 11 characters to 8 + "..."
    names = [name[:5] + "..." if len(name) > 11 else name for name in criterion.names]
    key = "weighted_binary_cross_entropy"
    if key in criterion.names:
        names[criterion.names.index(key)] = "WBCE"
    key = "binary_cross_entropy"
    if key in criterion.names:
        names[criterion.names.index(key)] = "BCE"

    print(("\n" + "%11s" * (5 + n_objectives)) % ("Epoch", "GPU_mem", "Loss", *names, "Instances", "Size"))
    mean_losses = torch.zeros(n_objectives, device=device)


    progress = tqdm.tqdm(enumerate(loader), total=len(loader))
    for i, (imgs, targets) in progress:
        imgs = imgs.to(device, non_blocking=True).float()
        optimizer.zero_grad()
        # Forward

        preds = model(imgs)  # (B, C, H, W) where C = number of classes
        preds = F.interpolate(preds, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        aggr_loss, losses = criterion(preds, targets.to(device, dtype=torch.float32))
        aggr_loss.backward()

        optimizer.step()
        running_loss = (running_loss * i + aggr_loss.item()) / (i + 1)
        mean_losses = (mean_losses * i + losses) / (i + 1)
        mem = f"{device_memory_used(device):.3g}G"

        progress.set_description(
            ("%11s" * 2 + "%11.4g" * (n_objectives + 3))
            % (
                f"{epoch + 1}/{hyp.epochs}",
                mem,
                running_loss,
                *mean_losses.tolist(),
                targets.shape[0],
                imgs.shape[-1],
            )
        )

    metrics = {f"train/{name}": cl.item() for name, cl in zip(criterion.names, mean_losses)}
    metrics["train/loss"] = running_loss
    return metrics


def postprocess_predictions(preds: torch.Tensor, conf: float = 0.5):
    B, C, H, W = preds.shape
    preds = torch.argmax(preds, dim=1) if C != 1 else F.sigmoid(preds) > conf
    return preds.to(torch.uint8)


@torch.inference_mode()
def valid_epoch(model: nn.Module, hyp, loader, epoch, criterion, device):
    running_loss = 0.0

    n_objectives = len(criterion)
    print(("\n" + "%11s" * 4) % ("Precision", "Recall", "Accuracy", "F1"))
    mean_losses = torch.zeros(n_objectives, device=device)

    confmat = BinaryConfusionMatrix()
    confmat.to(device)

    progress = tqdm.tqdm(enumerate(loader), total=len(loader))
    for i, (imgs, targets) in progress:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(
            device, non_blocking=True, dtype=torch.float32
        )  # TODO: remove dtype
        preds = model(imgs)  # (B, C, H, W) where C = number of classes
        preds = F.interpolate(preds, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        aggr_loss, losses = criterion(preds, targets)
        mask = postprocess_predictions(preds, conf=hyp.conf)
        targets = targets.long()
        confmat(mask, targets)
        running_loss = (running_loss * i + aggr_loss.item()) / (i + 1)
        mean_losses = (mean_losses * i + losses.detach()) / (i + 1)

        # update description with conf matrix
        progress.set_description(("%11.4g" * 4) % tuple(confmat.metrics().values()))

    metrics = {"valid/loss": running_loss }
    metrics = {**metrics, **confmat.metrics(prefix="valid/")}
    for name, loss in zip(criterion.names, mean_losses):
        metrics[f"valid/{name}"] = loss.item()

    return metrics


def auto_naming(hyp):
    """
    Automatically generate a name for the training run based on hyperparameters.
    """
    name = f"model_{hyp.model}_dataset_{hyp.dataset.name}_imgsz_{hyp.image_sz}_criterion_{hyp.criterion}"
    if hyp.pretrained:
        name += "_pretrained" if not hyp.resume else "_resumed"

    print(f"Auto-generated run name: {name}")
    hyp.name = name
    return hyp


def train(hyp: IterableSimpleNamespace, tracker: Tracker = Tracker):
    init_seeds(hyp.seed, deterministic=hyp.deterministic)
    hyp.pretrained = False
    hyp.dataset = Path(hyp.dataset)
    data = dataset_read_config(hyp.dataset / "config.yaml")  # dataset description

    model = init_model(hyp.model, data, hyp)
    save_dir = Path(hyp.save_dir)
    
    device = torch.device(hyp.device)

    # Check pretrained and resume
    weights: Path = Path(hyp.weights) if hyp.weights else None
    hyp.pretrained = weights is not None and weights.exists()
    hyp.resume = hyp.resume and hyp.pretrained

    if hyp.name is None:
        hyp = auto_naming(hyp)

    if hyp.tracker == "wandb":
        tracker = WandbTracker(project=hyp.project, name=hyp.name, config=vars(hyp))
    else:
        tracker = Tracker(hyp)
    nc = data.get("nc", 1)
    model.nc = nc

    # Use no extra workers for CPU/MPS devices.
    workers = 0 if device.type in {"cpu", "mps"} else hyp.workers
    criterion = AutoCriterion(hyp.criterion, model, hyp, data, device)

    # Define optimization components
    optimizer = torch.optim.Adam(model.parameters(), lr=hyp.lr, weight_decay=hyp.weight_decay)
    mean, std = data["mean"], data["std"]
    
    train_set = LandslideDataset(
        data["train"],
        image_sz=hyp.image_sz,
        mask_sz=hyp.mask_sz,
        mean=mean,
        std=std,
        do_normalize=hyp.normalize,
    )
    valid_set = LandslideDataset(
        data[hyp.val],
        image_sz=hyp.image_sz,
        mask_sz=hyp.mask_sz,
        mean=mean,
        std=std,
        do_normalize=hyp.normalize,
    )

    print(
        f"Training on {len(train_set)} samples with imgsz {hyp.image_sz} "
        f"and validating on {len(valid_set)} samples."
    )
    train_loader = dataloader(train_set, hyp.batch, workers, hyp.image_sz, mode="train")
    valid_loader = dataloader(valid_set, hyp.batch, workers, hyp.image_sz, mode="valid")

    start_epoch = 0

    weights_dir = save_dir / hyp.name / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    fitness = float("-inf") if hyp.mode == "max" else float("inf")
    best_epoch = 0


    if hyp.pretrained and not hyp.resume:
        model = load_model(model, weights, verbose=True)
        
    if hyp.resume:
        checkpoint = torch.load(weights, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        if "metrics" in checkpoint and hyp.monitor in checkpoint["metrics"]:
            fitness = checkpoint["metrics"][hyp.monitor]
        best_epoch = start_epoch
        print(f"Resuming training from epoch {start_epoch}")

    model = model.to(device)
    for epoch in range(start_epoch, hyp.epochs):
        print(f"Epoch {epoch+1}/{hyp.epochs}")
        # If you want training metrics (in addition to loss) pass compute_metrics=True.
        model.train()
        train_metrics = train_epoch(
            model, hyp, train_loader, epoch, criterion, device, optimizer
        )
        model.eval()
        valid_metrics = valid_epoch(model, hyp, valid_loader, epoch, criterion, device)
        device_memory_clear(device)
        gc.collect()
        metrics = {**train_metrics, **valid_metrics}
        tracker.log(metrics, step=epoch)
        def cmp(x, y):
            return x >= y if hyp.mode == "max" else x <= y
        if cmp(metrics[hyp.monitor], fitness):
            fitness = metrics[hyp.monitor]
            best_epoch = epoch
        model_checkpointing(
            model, optimizer, epoch, metrics, hyp, weights_dir, best_epoch, tracker
        )
        if (epoch - best_epoch) == hyp.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Visualize validation predictions at the end of training
    if hyp.tracker == "wandb" and _WANDB_AVAILABLE:
        import wandb
        print("Generating validation visualizations for Wandb...")
        model.eval()
        table = wandb.Table(columns=["Image", "Ground Truth", "Prediction"])
        mean = torch.tensor(data["mean"]).view(3, 1, 1).to(device)
        std = torch.tensor(data["std"]).view(3, 1, 1).to(device)

        for imgs, targets in valid_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            with torch.inference_mode():
                preds = model(imgs)
                preds = postprocess_predictions(preds, conf=hyp.conf)

            imgs = imgs * std + mean
            imgs = imgs * 255
            imgs = imgs.to(torch.uint8)
            
            for img, target, pred in zip(imgs, targets, preds):
                gt_mask = target.squeeze().cpu().numpy().astype(np.uint8) * 255
                gt_image = img.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                pred_mask = pred.squeeze().cpu().numpy().astype(np.uint8) * 255

                table.add_data(
                    wandb.Image(gt_image, caption="Input Image"),
                    wandb.Image(gt_mask, caption="Ground Truth"),
                    wandb.Image(pred_mask, caption="Prediction"),
                )

        # Log to wandb
        tracker.log({"Validation Samples": table})
        tracker.finish()
        print("Validation visualizations complete")
    return model


def model_checkpointing(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    hyp: dict,
    save_dir: Path,
    best_epoch: int = 0,
    tracker: Tracker = None,
):
    buffer = io.BytesIO()
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "hyp": hyp,
            "date": datetime.now().isoformat(),
        },
        buffer,
    )
    ckpt = buffer.getvalue()

    # last
    aliases = ["last"]
    last = save_dir / "last.pth"
    last.write_bytes(ckpt)

    if epoch == best_epoch:
        best = save_dir / "best.pth"
        best.write_bytes(ckpt)
        aliases.append("best")

    if (hyp.save_period > 0) and (epoch % hyp.save_period == 0):
        # save epoch, i.e. 'epoch_3.pt'
        (save_dir / f"epoch_{epoch}.pt").write_bytes(ckpt)
        # add alias (Currently disabled to save space)
        # aliases.append(f"epoch_{epoch + 1}")

    if tracker:
        tracker.log_model(last, aliases=aliases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model on landslide imagery."  )
    # Model and dataset
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "segformer"], help="Name of the model architecture to use (e.g., 'unet', 'fcn').")
    parser.add_argument("--project", type=str, default="landslide", help="Project name for tracking and logging.")
    parser.add_argument("--name", type=str, default=None, help="Name of the training run.")
    parser.add_argument("--dataset", type=str, default="L4S", help="Identifier or path for the dataset configuration.")
    parser.add_argument("--weights", type=str, default=None, help="Path to pretrained model weights to initialize training.")
    parser.add_argument("--resume", action="store_true", help="Whether to resume training from existing weights checkpoint.")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Total number of training epochs."
    )
    parser.add_argument("--batch", type=int, default=32, help="Batch size for training and validation.")
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
    hyp = IterableSimpleNamespace(**vars(args))

    train(hyp)