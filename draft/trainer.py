from dataclasses import dataclass
from datetime import datetime
import gc
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import tqdm

from landslide.data import LandslideDataset, dataloader, parse_dataset
from landslide.dtypes import IterableSimpleNamespace
from landslide.model import UNet
from landslide.torch_utils import (
    device_memory_clear,
    device_memory_used,
    init_seeds,
    select_device,
)
from landslide.utils import yaml_load

# Configure logger
logger = logging.getLogger(__name__)

# logging.basicConfig(level=logging.INFO)



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


class ModelWrapper:
    def __init__(self, model: str, config: dict):
        self.trainer = None
        pass
    
    def training_step(self, batch, batch_idx):
        pass
    def validation_step(self, batch, batch_idx):
        pass
    def optimization_step(self, loss):
        pass
    def forward(self, x):
        pass
    def preprocess(self, x):
        pass
    def postprocess(self, x):
        pass


class Trainer:
    config: TrainConfig
    optimizer: torch.optim.Optimizer = None
    scheduler: torch.optim.lr_scheduler._LRScheduler = None
    criterion: nn.Module | Callable = None
    device: torch.device = None
    model: nn.Module | nn.parallel.DataParallel = None
    callbacks: list[Callable] = []
    metrics: dict[str, float | int | str] = {}
    epoch: int = 0
    max_epochs: int = 100
    save_dir: Path = Path('.')
    data: dict = {}
    best_epoch: int = 0
    train_loader: torch.utils.data.DataLoader = None
    valid_loader: torch.utils.data.DataLoader = None
    start_epoch: int = 0
    stop: bool = False

    @property
    def weights_dir(self):
        return self.save_dir / 'weights'

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = select_device(config.device)
        self.max_epochs = config.epochs
        self.save_dir = Path(config.save_dir)
        # self.model = AutoModel(model, config)
        # self.criterion = AutoCriterion(model, config)
        # self.optimizer = AutoOptimizer(model, config)
        # self.scheduler = AutoScheduler(model, config)

    def run_callbacks(self, event: str):
        for callback in self.callbacks[event]:
            callback(self)

    def clear_memory(self):
        device_memory_clear(self.device)
        gc.collect()

    def log(
        self,
        name: str,
        value: Any,
        prog_bar: bool = False,
        logger: Optional[bool] = None,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx: Union[str, Callable] = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: Optional[Any] = None,
        add_dataloader_idx: bool = True,
        batch_size: Optional[int] = None,
        metric_attribute: Optional[str] = None,
        rank_zero_only: bool = False,
    ) -> None:
        """Log a key, value pair."""
        pass
    

    def train_epoch(self, model: ModelWrapper, loader: DataLoader, epoch: int):
        model.train()
        for idx, batch in enumerate(loader):
            batch = model.preprocess(batch)
            loss = model.training_step(batch, idx)
            model.optimization_step(loss)
    
    def valid_epoch(self, model: ModelWrapper, loader: DataLoader, epoch: int):
        model.eval()
        for idx, batch in enumerate(loader):
            batch = model.preprocess(batch)
            loss = model.validation_step(batch, idx)
        return loss


    def fit(self, model: ModelWrapper, train_set: Dataset, valid_set: Dataset, tracker = None):
        config: TrainConfig = self.config
        self.run_callbacks('on_pretrain_routine_start')
        init_seeds(config.seed)
        self.train_loader = dataloader(train_set, batch_size=config.batch_size, workers=config.workers)
        self.valid_loader = dataloader(valid_set, batch_size=config.batch_size, workers=config.workers)
        self.max_epochs = config.epochs
        self.run_callbacks('on_pretrain_routine_end')
        self.run_callbacks('on_train_start')
        for epoch in range(config.epochs):
            self.run_callbacks('on_epoch_start')
            model.train()
            for idx, batch in enumerate(self.train_loader):
                batch = model.preprocess(batch)
                loss = model.training_step(batch, idx)
                model.optimization_step(loss)
            model.eval()
            self.run_callbacks('on_epoch_fit')
            with torch.inference_mode():
                for idx, batch in enumerate(self.valid_loader):
                    batch = model.preprocess(batch)
                    loss = model.validation_step(batch, idx)
            self.clear_memory()
            self.run_callbacks('on_epoch_end')
        self.clear_memory()
        self.run_callbacks('on_train_end')
        return self.metrics
        

 