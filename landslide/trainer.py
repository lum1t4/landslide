from datetime import datetime
import io
from pathlib import Path
from typing import Callable, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import logging

from landslide.model import UNet
from landslide.torch import init_seeds, select_device
from landslide.data import LandslideDataset, dataloader, load_dataset
from landslide.dtypes import IterableSimpleNamespace
from landslide.utils import yaml_load
from landslide.trackers import Tracker, WandbTracker

# Configure logger
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)




class TrainContext:
    hyp: IterableSimpleNamespace
    device: torch.device
    criterion: nn.Module | Callable
    optimizer: torch.optim.Optimizer
    model: nn.Module
    train_loader: torch.utils.data.DataLoader
    valid_loader: Optional[torch.utils.data.DataLoader]
    test_loader: Optional[torch.utils.data.DataLoader]
    start_epoch: int = 0
    best_epoch: int
    epochs: int
    save_dir: Path
    batch_size: int
    tracker: Optional[Tracker]
    callbacks: List[Callable]
    metrics: dict
    workers: int = 8
    fitness: Optional[float | int]
    pretrained: bool = False
    resume: bool = False

    @property
    def weights_dir(self):
        return self.save_dir / self.hyp.name / "weights"
    
    @property
    def best(self):
        return self.weights_dir / "best.pth"
    
    @property
    def last(self):
        return self.weights_dir / "last.pth"
    

    def training_step(self, index, batch):
        raise NotImplementedError
    
    def validation_step(self, index, batch):
        raise NotImplementedError
    

    def train(self, loader, *args, **kwargs):
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            self.train_epoch(loader, epoch, *args, **kwargs)
            self.model.eval()
            self.valid_epoch(loader, epoch, *args, **kwargs)
            self.save_checkpoint(epoch)
            self.update_metrics(epoch)
            self.check_early_stopping(epoch)
            self.log_metrics(epoch)
            self.log_model(epoch)
            self.run_callbacks(epoch)







    



def pretrain_routine(ctx: TrainContext):
    init_seeds(ctx.hyp.seed, deterministic=ctx.hyp.deterministic)
    
    ctx.device = torch.device(ctx.hyp.device)
    
    ctx.model = load_model(ctx.hyp.model, ctx.data, ctx.hyp)
    ctx.model.to(ctx.device)

    ctx.weights = Path(ctx.hyp.weights) if ctx.hyp.weights else None

    ctx.pretrained = ctx.weights and ctx.weights.exists()
    ctx.hyp.resume = ctx.hyp.resume and ctx.pretrained
    # Use no extra workers for CPU/MPS devices.
    ctx.workers = 0 if ctx.device.type in {"cpu", "mps"} else hyp.workers


def rename_run(ctx: TrainContext):
    hyp.name = f"{hyp.model}_{hyp.dataset}_{hyp.image_sz}_{hyp.batch}_{hyp.lr}"    
    if ctx.pretrained:
        hyp.name += "_pretrained" if not hyp.resume else "_resumed"


def build_criterion(ctx: TrainContext):
    if ctx.hyp.criterion == "weighted_binary_cross_entropy":
        pos_weight = torch.tensor(ctx.data['pos_weights']).reshape(ctx.nc, 1, 1).to(ctx.device)
        ctx.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        ctx.criterion = nn.BCEWithLogitsLoss()


def build_optimizer(ctx: TrainContext):
    ctx.optimizer = torch.optim.Adam(ctx.model.parameters(), lr=ctx.hyp.lr, weight_decay=ctx.hyp.weight_decay)



    



def train_epoch(model, hyp, loader, epoch, criterion, device, optimizer):
    running_loss = 0.0

    progress = tqdm.tqdm(enumerate(loader), total=len(loader), desc='Training')
    for i, (img, labels) in progress:
        img = img.to(device, non_blocking=True)
        optimizer.zero_grad()
        # Forward
        preds = model(img) # (B, C, H, W) where C = number of classes
        if preds.shape[-2:] != labels.shape[-2:]:
            preds = nn.functional.interpolate(preds, size=labels.shape[-2:], mode="bilinear", align_corners=False)

        loss = criterion(preds, labels.to(device, dtype=torch.float32))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return {"train/loss": running_loss / len(loader)}


def postprocess(preds: torch.Tensor, hyp: IterableSimpleNamespace):
    _, ch, h, w = preds.shape
    preds = (F.sigmoid(preds) > hyp.conf) if ch == 1 else  torch.argmax(preds, dim=1)
    if h != hyp.image_sz or w != hyp.image_sz:
        preds = F.interpolate(preds, size=(hyp.image_sz, hyp.image_sz), mode="bilinear", align_corners=False)
    return preds.to(torch.uint8)


@torch.inference_mode()
def valid_epoch(model: nn.Module, hyp, loader, epoch, criterion, device):
    running_loss = 0.0
    tp, fp, fn, tn = 0.0, 0.0, 0.0, 0.0
    progress = tqdm.tqdm(enumerate(loader), total=len(loader), desc='Validation')
    for i, (img, labels) in progress:
        img = img.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True, dtype=torch.float32) # TODO: remove dtype
        preds = model(img) # (B, C, H, W) where C = number of classes
        if preds.shape[-2:] != labels.shape[-2:]:
            preds = F.interpolate(preds, size=labels.shape[-2:], mode="bilinear", align_corners=False)
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
    accuracy = (tp +  tn) / (tp + tn + fp + fn + epsilon)
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


def train(model, hyp, data, save_dir, tracker: Tracker = Tracker):
    init_seeds(hyp.seed, deterministic=hyp.deterministic)
    pretrained = False

    device = torch.device(hyp.device)
    pretrained = hyp.weights is not None and Path(hyp.weights).exists()
    hyp.resume = hyp.resume and pretrained


    hyp.name = f"{hyp.model}_{hyp.dataset}_{hyp.image_sz}_{hyp.batch}_{hyp.lr}"    
    if pretrained:
        hyp.name += "_pretrained" if not hyp.resume else "_resumed"

    
    tracker = Tracker(hyp)
    nc = data.get('nc', 1)
    model.nc = nc
    model.to(device)

    # Use no extra workers for CPU/MPS devices.
    workers = 0 if device.type in {"cpu", "mps"} else hyp.workers


    if hyp.criterion == "weighted_binary_cross_entropy":
        pos_weight = torch.tensor(data['pos_weights']).reshape(nc, 1, 1).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()


    optimizer = torch.optim.Adam(model.parameters(), lr=hyp.lr, weight_decay=hyp.weight_decay)
    
    train_set = LandslideDataset(
        data['train'],
        mean=data['mean'],
        std=data['std'],
        image_sz=hyp.image_sz,
        mask_sz=hyp.mask_sz,
        do_normalize=True,
    )
    valid_set = LandslideDataset(
        data[hyp.val],
        mean=data['mean'],
        std=data['std'],
        image_sz=hyp.image_sz,
        mask_sz=hyp.mask_sz,
        do_normalize=True,
        do_rescale=True,
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

    if hyp.resume:
        checkpoint = torch.load(hyp.weights)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        if "metrics" in checkpoint and hyp.monitor in checkpoint["metrics"]:
            fitness = checkpoint["metrics"][hyp.monitor]
        best_epoch = start_epoch
        logger.info(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, hyp.epochs):
        logger.info(f"Epoch {epoch+1}/{hyp.epochs}")
        # If you want training metrics (in addition to loss) pass compute_metrics=True.
        model.train()
        train_metrics = train_epoch(model, hyp, train_loader, epoch, criterion, device, optimizer)

        model.eval()
        valid_metrics = valid_epoch(model, hyp, valid_loader, epoch, criterion, device)        
        metrics = {**train_metrics, **valid_metrics}
        tracker.log(metrics, step=epoch)

        def cmp(x, y):
            return x >= y if hyp.mode == 'max' else x <= y

        if cmp(metrics[hyp.monitor], fitness):
            fitness = metrics[hyp.monitor]
            best_epoch = epoch

        model_checkpointing(model, optimizer, epoch, metrics, hyp, weights_dir, best_epoch, tracker)

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
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "hyp": hyp,
        "date": datetime.now().isoformat(),
    }, buffer)
    ckpt = buffer.getvalue()

    # last
    last = save_dir / "last.pth"
    last.write_bytes(ckpt)

    if epoch == best_epoch:
        best = save_dir / "best.pth"
        best.write_bytes(ckpt)


    if (hyp.save_period > 0) and (epoch % hyp.save_period == 0):
            (save_dir / f"epoch_{epoch}.pt").write_bytes(ckpt)  # save epoch, i.e. 'epoch3.pt'

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
        model = "unet",
        project = "landslide",
        dataset = "A19",
        name = None,
        weights = None, # model weights if using a pretrained model
        resume = False,
        image_sz = 128,
        mask_sz = 128,
        conf = 0.5,
        seed = 1337,
        save_period = -1,
        deterministic = True,
        batch = 32,
        workers = 8,
        monitor = "valid/F1",
        patience = 10,
        mode = "max",
        val = "valid",
        weight_decay = 5e-4,
        ignore_index = None, # or 255
        criterion = "weighted_binary_cross_entropy",
        epochs = 100,
        normalize = True, # not yet used
        lr = 1e-3,
        device = "mps:0",
    )

    hyp = IterableSimpleNamespace(**hyp)
    data = load_dataset(hyp.dataset) # dataset description
    model = load_model(hyp.model, data, hyp)

    train(model, hyp, data, save_dir=Path("./runs"))

