from datetime import datetime
import io
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from landslide.data import LandslideDataset, dataloader, load_dataset
from landslide.dtypes import IterableSimpleNamespace
from landslide.losses import BinaryFocalLossWithLogits, DiceLoss, LovaszHingeLoss
from landslide.model import UNet
from landslide.torch import device_memory_used, init_seeds
from landslide.trackers import Tracker

# Configure logger
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


def train_epoch(model, hyp, loader, epoch, criterion, device, optimizer):
    running_loss = 0.0

    s = ("\n" + "%11s" * 5) % ("Epoch", "GPU_mem", "loss", "Instances", "Size")
    print(s)
    progress = tqdm.tqdm(enumerate(loader), total=len(loader))
    for i, (imgs, targets) in progress:
        imgs = imgs.to(device, non_blocking=True)
        optimizer.zero_grad()
        # Forward
        preds = model(imgs)  # (B, C, H, W) where C = number of classes
        if preds.shape[-2:] != targets.shape[-2:]:
            preds = nn.functional.interpolate(
                preds, size=targets.shape[-2:], mode="bilinear", align_corners=False
            )

        loss = criterion(preds, targets.to(device, dtype=torch.float32))
        loss.backward()
        optimizer.step()

        running_loss = (running_loss * i + loss.item()) / (i + 1)
        mem = f"{device_memory_used(device):.3g}G"

        progress.set_description(
            ("%11s" * 2 + "%11.4g" * 3)
            % (
                f"{epoch + 1}/{hyp.epochs}",
                mem,
                running_loss,
                targets.shape[0],
                imgs.shape[-1],
            )
        )

    return {"train/loss": running_loss}


def postprocess(preds: torch.Tensor, hyp: IterableSimpleNamespace):
    _, ch, h, w = preds.shape
    preds = (F.sigmoid(preds) > hyp.conf) if ch == 1 else torch.argmax(preds, dim=1)
    if h != hyp.image_sz or w != hyp.image_sz:
        preds = F.interpolate(
            preds,
            size=(hyp.image_sz, hyp.image_sz),
            mode="bilinear",
            align_corners=False,
        )
    return preds.to(torch.uint8)


@torch.inference_mode()
def valid_epoch(model: nn.Module, hyp, loader, epoch, criterion, device):
    running_loss = 0.0
    tp, fp, fn, tn = 0.0, 0.0, 0.0, 0.0
    progress = tqdm.tqdm(enumerate(loader), total=len(loader), desc="Validation")
    for i, (img, labels) in progress:
        img = img.to(device, non_blocking=True)
        labels = labels.to(
            device, non_blocking=True, dtype=torch.float32
        )  # TODO: remove dtype
        preds = model(img)  # (B, C, H, W) where C = number of classes
        if preds.shape[-2:] != labels.shape[-2:]:
            preds = F.interpolate(
                preds, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
        loss = criterion(preds, labels)
        running_loss += loss.item()
        mask = postprocess(preds, hyp)
        labels = labels.to(torch.uint8)
        tp += ((mask == 1) & (labels == 1)).sum().item()
        fp += ((mask == 1) & (labels == 0)).sum().item()
        fn += ((mask == 0) & (labels == 1)).sum().item()
        tn += ((mask == 0) & (labels == 0)).sum().item()

    avg_loss = running_loss / len(loader)
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)

    print(f"conf matrix: tp={tp}, fp={fp}, fn={fn}, tn={tn}")
    metrics = {
        "valid/loss": avg_loss,
        "valid/Precision": precision,
        "valid/Recall": recall,
        "valid/F1": f1,
        "valid/IoU": iou,
        "valid/Accuracy": accuracy,
    }
    return metrics


def build_criterion(model, hyp, data, device):
    nc = data.get("nc", 1)
    match hyp.criterion:
        case "binary_cross_entropy":
            return nn.BCEWithLogitsLoss()
        case "focal_loss":
            return BinaryFocalLossWithLogits(alpha=0.25, gamma=2.0)
        case "lovasz_hinge_loss":
            return LovaszHingeLoss()
        case "weighted_binary_cross_entropy":
            pos_weight = torch.tensor(data["pos_weights"]).reshape(nc, 1, 1).to(device)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        case "lovasz_loss":
            return LovaszHingeLoss()
        case "dice_loss":
            return DiceLoss()
        case _:
            logger.warning(
                f"Unknown criterion: {hyp.criterion}, using BCEWithLogitsLoss instead"
            )
            return nn.BCEWithLogitsLoss()


def train(model, hyp, data, save_dir, tracker: Tracker = Tracker):
    init_seeds(hyp.seed, deterministic=hyp.deterministic)
    pretrained = False

    device = torch.device(hyp.device)

    # Check pretrained and resume
    weights = Path(hyp.weights) if hyp.weights else None
    pretrained = weights and weights.exists()
    hyp.resume = hyp.resume and pretrained

    # Rename run based on hyperparameters
    hyp.name = f"{hyp.model}_{hyp.dataset}_{hyp.image_sz}_{hyp.batch}_{hyp.lr}"
    if pretrained:
        hyp.name += "_pretrained" if not hyp.resume else "_resumed"

    logger.info(f"Run: f{hyp.name}")
    tracker = Tracker(hyp)
    nc = data.get("nc", 1)
    model.nc = nc

    # Use no extra workers for CPU/MPS devices.
    workers = 0 if device.type in {"cpu", "mps"} else hyp.workers

    criterion = build_criterion(model, hyp, data, device)

    # Define optimization components
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyp.lr, weight_decay=hyp.weight_decay
    )

    train_set = LandslideDataset(
        data["train"],
        mean=data["mean"],
        std=data["std"],
        image_sz=hyp.image_sz,
        mask_sz=hyp.mask_sz,
        do_normalize=True,
    )
    valid_set = LandslideDataset(
        data[hyp.val],
        mean=data["mean"],
        std=data["std"],
        image_sz=hyp.image_sz,
        mask_sz=hyp.mask_sz,
        do_normalize=True,
        do_rescale=True,
    )

    logger.info(
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

    if hyp.resume:
        checkpoint = torch.load(weights, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        if "metrics" in checkpoint and hyp.monitor in checkpoint["metrics"]:
            fitness = checkpoint["metrics"][hyp.monitor]
        best_epoch = start_epoch
        logger.info(f"Resuming training from epoch {start_epoch}")

    model = model.to(device)
    for epoch in range(start_epoch, hyp.epochs):
        logger.info(f"Epoch {epoch+1}/{hyp.epochs}")
        # If you want training metrics (in addition to loss) pass compute_metrics=True.
        model.train()
        train_metrics = train_epoch(
            model, hyp, train_loader, epoch, criterion, device, optimizer
        )

        model.eval()
        valid_metrics = valid_epoch(model, hyp, valid_loader, epoch, criterion, device)
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
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

        logger.info(f"Epoch {epoch+1} metrics: {metrics}")

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
    last = save_dir / "last.pth"
    last.write_bytes(ckpt)

    if epoch == best_epoch:
        best = save_dir / "best.pth"
        best.write_bytes(ckpt)

    if (hyp.save_period > 0) and (epoch % hyp.save_period == 0):
        (save_dir / f"epoch_{epoch}.pt").write_bytes(
            ckpt
        )  # save epoch, i.e. 'epoch_3.pt'

    if tracker:
        aliases = ["last", f"epoch_{epoch+1}"]
        if epoch == best_epoch:
            aliases.append("best")
        tracker.log_model(last, aliases=aliases)


def load_model(model: str, data: dict, hyp: IterableSimpleNamespace):
    # Load the model
    # TODO: Implement model loading
    return UNet(nc=1, ch=3)


if __name__ == "__main__":
    hyp = dict(
        model="unet",
        project="landslide",
        dataset="A19",
        name=None,
        weights=None,  # model weights if using a pretrained model
        resume=False,
        image_sz=128,
        mask_sz=128,
        conf=0.5,
        seed=1337,
        save_period=-1,
        deterministic=True,
        batch=32,
        workers=8,
        monitor="valid/F1",
        patience=10,
        mode="max",
        val="valid",
        weight_decay=5e-4,
        ignore_index=None,  # or 255
        criterion="weighted_binary_cross_entropy",
        epochs=100,
        normalize=True,  # not yet used
        lr=1e-3,
        device="mps:0",
        tracker="wandb",
    )

    hyp = IterableSimpleNamespace(**hyp)
    data = load_dataset(hyp.dataset)  # dataset description
    model = load_model(hyp.model, data, hyp)

    train(model, hyp, data, save_dir=Path("./runs"))
