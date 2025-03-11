import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from landslide.torch import init_seeds, device_memory_used
from landslide.data import LS3, dataloader


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, nc: int = 1, ch: int = 14, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = ch
        self.n_classes = 1  # one output channel for binary segmentation
        self.bilinear = bilinear

        self.inc = DoubleConv(ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, nc)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# -----------------------
# Hyperparameters & Configurations
# -----------------------
imgsz = 128
ch = 3
batch_size = 64
workers = 8
weight_decay = 5e-4
epochs = 50
lr = 1e-3
start_epoch = 0
epsilon = 1e-14  # for numerical stability



# -----------------------
# Training and validation loops
# -----------------------
def train_epoch(model, loader, epoch, criterion, device, optimizer, compute_metrics=False):
    model.train()
    running_loss = 0.0
    # For optional metric computation on the training set
    total_tp, total_fp, total_fn = 0.0, 0.0, 0.0
    
    progress = tqdm.tqdm(enumerate(loader), total=len(loader), desc='Training')
    for i, (img, target) in progress:
        progress.set_description(f"Training (epoch {epoch+1}, memory used: {device_memory_used(device):.3f}GB)")
        img = img.to(device, non_blocking=True)
        optimizer.zero_grad()
        preds = model(img).squeeze(1)  # (B, 1, H, W) -> (B, H, W) if NC = 1
        loss = criterion(preds, target.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if compute_metrics:
            # Get predicted labels using argmax. Note: no softmax is needed for argmax.
            pred_labels = preds.argmax(dim=1)
            total_tp += ((pred_labels == 1) & (target == 1)).sum().item()
            total_fp += ((pred_labels == 1) & (target == 0)).sum().item()
            total_fn += ((pred_labels == 0) & (target == 1)).sum().item()
    
    avg_loss = running_loss / len(loader)
    metrics = {"train/loss": avg_loss}
    if compute_metrics:
        precision = total_tp / (total_tp + total_fp + epsilon)
        recall = total_tp / (total_tp + total_fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        metrics.update({"train/precision": precision, "train/recall": recall, "train/f1": f1})
    return metrics


@torch.inference_mode()
def valid_epoch(model, loader, epoch, criterion, device):
    model.eval()
    running_loss = 0.0
    tp, fp, fn = 0.0, 0.0, 0.0
    
    progress = tqdm.tqdm(enumerate(loader), total=len(loader), desc='Validation')
    for i, (img, target) in progress:
        img = img.to(device, non_blocking=True)
        preds = model(img).squeeze(1) # (B, 1, H, W) -> (B, H, W) if NC = 1
        target = target.to(device)
        loss = criterion(preds, target)
        running_loss += loss.item()
        
        preds = (torch.sigmoid(preds) > 0.5).long()
        target = target.long()
        tp += ((preds == 1) & (target == 1)).sum().item()
        fp += ((preds == 1) & (target == 0)).sum().item()
        fn += ((preds == 0) & (target == 1)).sum().item()
    
    avg_loss = running_loss / len(loader)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    metrics = {
        "valid/loss": avg_loss,
        "valid/precision": precision,
        "valid/recall": recall,
        "valid/f1": f1
    }
    return metrics

# -----------------------
# The main training function (modified)
# -----------------------
def train(m, hyp, data, device):
    init_seeds(1337, deterministic=True)
    # Initialize the UNet model.
    model = UNet(nc=1, ch=ch).to(device)
    # Optionally wrap your model to ensure outputs are resized to imgsz.
    # (Your UNet might output a different spatial size, so we upsample here.)
    model = nn.Sequential(model, nn.Upsample(size=(imgsz, imgsz), mode='bilinear', align_corners=True))
    
    # Use no extra workers for CPU/MPS devices.
    workers = 0 if device.type in {"cpu", "mps"} else workers
    
    # For binary segmentation with 2 output channels, CrossEntropyLoss is a good choice.
    # (If you wanted BCEWithLogitsLoss, you'd need to change your UNet to output 1 channel.)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_set = LS3('./dataset/phase_I/train')
    valid_set = LS3('./dataset/phase_I/valid')
    train_loader = dataloader(train_set, batch_size, workers, imgsz, mode="train")
    valid_loader = dataloader(valid_set, batch_size, workers, imgsz, mode="valid")
    
    for epoch in range(start_epoch, epochs):
        # If you want training metrics (in addition to loss) pass compute_metrics=True.
        train_metrics = train_epoch(model, train_loader, epoch, criterion, device, optimizer, compute_metrics=False)
        valid_metrics = valid_epoch(model, valid_loader, epoch, criterion, device)
        metrics = {**train_metrics, **valid_metrics}
        print(f"Epoch {epoch+1}: {metrics}")
    
    return model


if __name__ == "__main__":
    train(None, None, None, torch.device('mps'))