
from wansport.utils import IterableSimpleNamespace
from wansport.utils.torch import ModelEMA
from torch import nn
import torch
from pathlib import Path
from typing import Callable, Optional



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

