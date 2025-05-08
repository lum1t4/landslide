from pathlib import Path

import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torch
import torch.nn.functional as F

from landslide.data import LandslideDataset, dataloader, dataset_read_config
from landslide.dtypes import IterableSimpleNamespace
from landslide.losses import AutoCriterion
from landslide.metrics import BinaryConfusionMatrix
from landslide.model import load_model
from landslide.torch_utils import init_seeds, select_device


class AutoModel(L.LightningModule):
    def __init__(self, model, criterion: AutoCriterion, hyp, data):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.hyp = hyp
        self.data = data
        self.confmat = BinaryConfusionMatrix()
        self.save_hyperparameters(ignore=['model', 'criterion'])
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        preds = self(imgs)
        # Resize predictions if needed
        if preds.shape[-2:] != targets.shape[-2:]:
            preds = F.interpolate(preds, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        # Calculate loss
        aggr_loss, losses = self.criterion(preds, targets)
        
        # Log metrics
        self.log("train/loss", aggr_loss, prog_bar=True, on_step=True, on_epoch=True)
        for i, name in enumerate(self.criterion.names):
            self.log(f"train/{name}", losses[i], on_epoch=True)
            
        return aggr_loss
    
    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        preds = self(imgs)
        # Resize predictions if needed
        if preds.shape[-2:] != targets.shape[-2:]:
            preds = F.interpolate(preds, size=targets.shape[-2:], mode="bilinear", align_corners=False)
            
        # Calculate loss
        aggr_loss, losses = self.criterion(preds, targets)
        
        # Convert predictions to binary masks
        if preds.shape[1] == 1:
            mask = (F.sigmoid(preds) > self.hyp.conf).to(torch.uint8)
        else:
            mask = torch.argmax(preds, dim=1).to(torch.uint8)
            
        # Update confusion matrix
        self.confmat.update(mask, targets.long())
        # Log metrics
        self.log("valid/loss", aggr_loss, prog_bar=True, on_epoch=True)
        for i, name in enumerate(self.criterion.names):
            self.log(f"valid/{name}", losses[i], on_epoch=True)
    
        return {"loss": aggr_loss}
    
    def on_validation_epoch_end(self):
        metrics = self.confmat.metrics(prefix="valid/")
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True)
        
        # Reset confusion matrix for next validation
        self.confmat = BinaryConfusionMatrix()
        
        return metrics
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hyp.lr, 
            weight_decay=self.hyp.weight_decay
        )
        return optimizer


def train(hyp, tracker=None):
    init_seeds(hyp.seed, deterministic=hyp.deterministic)
    
    # Load data
    data = dataset_read_config(hyp.dataset)
    
    # Load model
    model = load_model(hyp.model, data, hyp)
    save_dir = Path(hyp.save_dir)
    
    # Check for pretrained weights
    weights = Path(hyp.weights) if hyp.weights else None
    pretrained = weights and weights.exists()
    resume = hyp.resume and pretrained
    
    # Rename run based on hyperparameters
    hyp.name = f"{hyp.model}_{hyp.dataset}_{hyp.image_sz}_{hyp.batch}_{hyp.lr}"
    if pretrained:
        hyp.name += "_pretrained" if not resume else "_resumed"
    
    # Initialize criterion
    device = torch.device(hyp.device)
    criterion = AutoCriterion(hyp.criterion, model, hyp, data, device)
    
    # Create Lightning module
    model = AutoModel(model, criterion, hyp, data)

    
    # Load checkpoint if resuming
    if resume:
        try:
            # Try loading as a Lightning checkpoint first
            lit_model = AutoModel.load_from_checkpoint(
                weights, 
                model=model, 
                criterion=criterion, 
                hyp=hyp, 
                data=data
            )
        except Exception as e:
            print(f"Could not load as Lightning checkpoint: {e}")
            # Load as regular torch state dict
            checkpoint = torch.load(weights, map_location="cpu")
            model.load_state_dict(checkpoint)
            lit_model = AutoModel(model, criterion, hyp, data)
    
    # Prepare datasets - using the same dataset classes as before
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
    
    # Create dataloaders
    workers = 0 if device.type in {"cpu", "mps"} else hyp.workers
    train_loader = dataloader(train_set, hyp.batch, workers, shuffle=True)
    valid_loader = dataloader(valid_set, hyp.batch, workers, shuffle=False)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir / hyp.name / "weights",
        filename="model-{epoch:02d}-{valid/F1-Score:.4f}",
        monitor=hyp.monitor,
        mode=hyp.mode,
        save_top_k=1,
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor=hyp.monitor,
        mode=hyp.mode,
        patience=hyp.patience,
    )
    
    # Set up logger
    logger = None
    if hyp.tracker == "wandb":
        logger = WandbLogger(project=hyp.project, name=hyp.name, config=vars(hyp))
    
    device = select_device(hyp.device)
    
    trainer = Trainer(
        max_epochs=hyp.epochs,
        accelerator=device,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
        deterministic=hyp.deterministic,
    )
    
    # Start training
    trainer.fit(lit_model, train_loader, valid_loader)
    
    return model


if __name__ == "__main__":
    hyp = dict(
        model="unet",
        project="landslide",
        dataset="A19",
        name=None,
        weights=None,
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
        patience=10,
        mode="max",
        val="valid",
        weight_decay=5e-4,
        ignore_index=None,
        criterion="weighted_binary_cross_entropy",
        epochs=100,
        normalize=True,
        lr=1e-3,
        device="mps:0",
        tracker="wandb",
        save_dir="./runs",
    )

    hyp = IterableSimpleNamespace(**hyp)
    train(hyp)