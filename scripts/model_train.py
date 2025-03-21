from datetime import datetime
import gc
import io
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from landslide.data import LandslideDataset, dataloader, parse_dataset
from landslide.dtypes import IterableSimpleNamespace
from landslide.losses import AutoCriterion
from landslide.metrics import BinaryConfusionMatrix
from landslide.model import load_model
from landslide.torch import device_memory_clear, device_memory_used, init_seeds
from landslide.trackers import Tracker, WandbTracker

# Configure logger
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


def train_epoch(model, hyp, loader, epoch, criterion: AutoCriterion, device, optimizer):
    
    running_loss = 0.0
    n_objectives = len(criterion)

    # DIRTY HACK: shorten the name of the criterion
    # trunacate the names longer than 11 characters to 8 + "..."
    names = [name[:5] + "..." if len(name) > 11 else name for name in criterion.names]
    key = "weighted_binary_cross_entropy"
    if key in criterion.names:
        names[criterion.names.index(key)] = "wbce"
        
    print(("\n" + "%11s" * (5 + n_objectives)) % ("Epoch", "GPU_mem", "Loss", *names, "Instances", "Size"))
    mean_losses = torch.zeros(n_objectives, device=device)


    progress = tqdm.tqdm(enumerate(loader), total=len(loader))
    for i, (imgs, targets) in progress:
        imgs = imgs.to(device, non_blocking=True).float()
        optimizer.zero_grad()
        # Forward
        preds = model(imgs)  # (B, C, H, W) where C = number of classes
        if preds.shape[-2:] != targets.shape[-2:]:
            preds = nn.functional.interpolate(
                preds, size=targets.shape[-2:], mode="bilinear", align_corners=False
            )

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

    return {f"train/loss_{name}": loss.item() for name, loss in zip(criterion.names, mean_losses)}


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
        if preds.shape[-2:] != targets.shape[-2:]:
            preds = F.interpolate(
                preds, size=targets.shape[-2:], mode="bilinear", align_corners=False
            )
        aggr_loss, losses = criterion(preds, targets)
        running_loss += aggr_loss.item()
        mask = postprocess(preds, hyp)
        targets = targets.long()
        confmat.update(mask, targets)
        running_loss = (running_loss * i + aggr_loss.item()) / (i + 1)
        mean_losses = (mean_losses * i + losses) / (i + 1)

        # update description with conf matrix
        progress.set_description(("%11.4g" * 4) % tuple(confmat.metrics().values()))

    metrics = {"valid/loss": running_loss / len(loader)}
    metrics = {**metrics, **confmat.metrics(prefix="valid/")}
    for name, loss in zip(criterion.names, mean_losses):
        metrics[f"valid/{name}"] = loss.item()

    return metrics



def train(hyp, tracker: Tracker = Tracker):
    init_seeds(hyp.seed, deterministic=hyp.deterministic)
    pretrained = False
    
    data = parse_dataset(hyp.dataset)  # dataset description
    model = load_model(hyp.model, data, hyp)
    save_dir = Path(hyp.save_dir)
    
    device = torch.device(hyp.device)

    # Check pretrained and resume
    weights = Path(hyp.weights) if hyp.weights else None
    pretrained = weights and weights.exists()
    hyp.resume = hyp.resume and pretrained

    # Rename run based on hyperparameters
    hyp.name = f"model_{hyp.model}_dataset_{hyp.dataset}_imagesz_{hyp.image_sz}_batch_{hyp.batch}_lr_{hyp.lr}"
    if pretrained:
        hyp.name += "_pretrained" if not hyp.resume else "_resumed"

    logger.info(f"Run: f{hyp.name}")

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
        monitor="valid/F1-Score",
        patience=5,
        mode="max",
        val="valid",
        weight_decay=5e-4,
        ignore_index=None,  # or 255
        criterion="weighted_binary_cross_entropy",
        epochs=30,
        normalize=True,  # not yet used
        lr=1e-3,
        device="mps:0",
        tracker="wandb",
        save_dir="./runs",
    )

    import argparse
    parser = argparse.ArgumentParser()
    for k, v in hyp.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    hyp = vars(parser.parse_args())
    hyp = IterableSimpleNamespace(**hyp)
    train(hyp)
