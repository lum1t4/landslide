
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn
from wansport.utils import IterableSimpleNamespace
from wansport.utils.torch import ModelEMA

TorchIO = torch.Tensor | tuple[torch.Tensor, ...] | List[torch.Tensor] | dict[str, torch.Tensor]


class TrainContext:
    model: nn.Module | nn.parallel.DistributedDataParallel
    data: dict
    hyp: IterableSimpleNamespace
    device: torch.device
    save_dir: Path
    callbacks: Optional[Callable] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
    model_ema: Optional[ModelEMA] = None
    scaler: Optional[torch.amp.GradScaler] = None
    criterion: Optional[Callable[[TorchIO, TorchIO], TorchIO]] = None
    start_epoch: int = 0
    epoch: int = 0
    best_epoch: int = 0
    metrics: dict = {}


    def train_epoch(model, data, hyp, loader, epoch, criterion, optimizer, device, scaler) -> tuple[nn.Module | nn.parallel.DistributedDataParallel, dict]:
        return model, {}
    
    def valid_epoch(model, data, hyp, loader, epoch, criterion, device) -> dict:
        return {}
    
    def train(model, hyp, data, device):
        model.to(device)
        hyp.workers = 0 if device.type in {"cpu", "mps"} else hyp.workers


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    workers: int = 8
    optimizer: str
    scheduler: Optional[str] = None
    criterion: Optional[str] = None
    device: str = "cpu"
    name: str = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    save_dir: Path = Path(".")
    seed: int = 0
    amp: bool = False