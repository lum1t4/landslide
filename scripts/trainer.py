from pathlib import Path
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import tqdm
import logging

from landslide.model import UNet
from landslide.torch import init_seeds
from landslide.data import LandslideDataset, dataloader, load_dataset
from landslide.dtypes import IterableSimpleNamespace
from landslide.utils import yaml_load


# Configure logger
logger = logging.getLogger(__name__)

# logging.basicConfig(level=logging.INFO)


class TrainContext:
    optimizer: torch.optim.Optimizer = None
    scheduler: torch.optim.lr_scheduler._LRScheduler = None
    criterion: nn.Module | Callable = None
    device: torch.device = None
    model: nn.Module | nn.parallel.DataParallel = None
    callbacks: list[Callable] = []
    metrics: dict[str, float | int | str] = {}
    epoch: int = 0
    max_epochs: int = 0
    save_dir: Path = Path('.')
    data: dict = {}
    best_epoch: int = 0
    train_loader: torch.utils.data.DataLoader = None
    valid_loader: torch.utils.data.DataLoader = None


class Trainer:
    optimizer: torch.optim.Optimizer = None
    scheduler: torch.optim.lr_scheduler._LRScheduler = None
    criterion: nn.Module | Callable = None
    device: torch.device = None
    model: nn.Module | nn.parallel.DataParallel = None
    callbacks: list[Callable] = []
    metrics: dict[str, float | int | str] = {}
    epoch: int = 0
    max_epochs: int = 0
    save_dir: Path = Path('.')
    data: dict = {}
    best_epoch: int = 0
    train_loader: torch.utils.data.DataLoader = None
    valid_loader: torch.utils.data.DataLoader = None

    def __init__(self, hyp):
        self.hyp

    def train(self, model: nn.Module | nn.parallel.DataParallel, train, valid):
        for epoch in range(self.hyp.epochs):
            self.optimizer.zero_grad()
            model.train()
            
            for batch_idx, batch in enumerate(train):
                self.train_step(model, batch, batch_idx)
            self.optimizer.step()
            self.scheduler.step()
            model.eval()
            for batch_idx, batch in enumerate(valid):
                pass
                # model.validation_step(batch, batch_idx)


    def train_epoch(self):
        progress = enumerate(self.train_loader)
        progress = tqdm.tqdm(progress, total=len(self.train_loader), desc='Training')
        running_loss = 0.0
        self.optimizer.zero_grad()
        for batch_idx, (img, labels) in progress:
            imgs = img.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            preds = self.model(imgs)
            loss = self.criterion(preds, labels.to(self.device, dtype=torch.float32))
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        self.log("train/loss", running_loss / len(self.train_loader))
        
    
    def build_model(model: str, data: dict, hyp: dict | IterableSimpleNamespace):
        pass
    
    def build_optimizer(model, loader, hyp):
        pass

    def build_scheduler():
        pass

    def build_criterion(model: nn.Module, hyp: dict | IterableSimpleNamespace):
        pass

    def log(self, key: str, value: float | int | str):
        self.metrics[key] = value



