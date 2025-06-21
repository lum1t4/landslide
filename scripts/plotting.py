from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torchmetrics.classification import AUROC

import wandb
from landslide.data import LandslideDataset, dataloader, dataset_read_config
from landslide.metrics import BinaryConfusionMatrix
from landslide.model import init_model
from landslide.torch_utils import intersect_dicts


def load_model(model: nn.Module, weights: Path, verbose: bool = True) -> nn.Module:
    from safetensors.torch import load_file

    checkpoint = (
        torch.load(weights, map_location="cpu", weights_only=False)
        if not weights.name.endswith(".safetensors")
        else load_file(weights, device="cpu")
    )
    checkpoint = checkpoint["model"] if "model" in checkpoint else checkpoint
    csd = intersect_dicts(checkpoint, model.state_dict())  # intersect
    model.load_state_dict(csd, strict=False)  # load
    if verbose:
        print(f"Transferred {len(csd)}/{len(model.state_dict())} items from pretrained weights")
    return model


def plot_f1_at_threshold(x: list, y: list):
    """
    Plot F1 score at different thresholds.
    :param x: List of thresholds.
    :param y: List of F1 scores corresponding to the thresholds.
    """
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs. Threshold")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def visualise_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    save_path: Path,
    mean: list[float],
    std: list[float],
    normalized: bool = True,
):
    """
    Save a grid (rows=batch, cols=[image | GT mask | prediction]).
    All tensors expected on CPU: 
      images  -> (B,3,H,W) float in [0,1] **after** normalisation
      masks   -> (B,1,H,W) {0,1}
      preds   -> (B,1,H,W) float in [0,1]
    """

    if normalized:
        # denormalise to [0,255] uint8
        _mean = torch.tensor(mean).view(3, 1, 1)
        _std  = torch.tensor(std).view(3, 1, 1)
        imgs  = (images * _std + _mean).clamp(0, 1) * 255
    imgs  = imgs.to(torch.uint8)

    # masks/preds to 3-ch uint8 so they plot like images
    def to_rgb(x):
        x = (x.squeeze(1) * 255).to(torch.uint8)          # (B,H,W)
        return x.unsqueeze(1).repeat(1, 3, 1, 1)           # (B,3,H,W)

    masks_rgb = to_rgb(masks)
    preds_rgb = to_rgb((preds > 0.5).float())              # hard-threshold for display

    # concat along batch dim: [img, gt, pred, img, gt, pred, ...]
    triplets = torch.cat([imgs, masks_rgb, preds_rgb], dim=0)

    # grid with nrow = batch_size (keeps rows aligned)
    grid = vutils.make_grid(triplets, nrow=imgs.size(0), padding=2)
    plt.figure(figsize=(imgs.size(0) * 3, 3))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()





@torch.inference_mode()
def run_test(model_name: str, dataset: Path, checkpoint: Path, epoch: int, device: str = "cpu", save_dir: Path = Path("predictions")) -> dict:
    auroc_metric = AUROC(task="binary")
    data = dataset_read_config(dataset / "config.yaml")
    model = init_model(model_name, data)
    model = load_model(model, checkpoint)
    device = torch.device(device)
    tresholds = torch.linspace(0., .95, 20)
    confmats = [BinaryConfusionMatrix(threshold=t) for t in tresholds]
    fold, mean, std = data["test"], data["mean"], data["std"]
    imgsz = 512 if model_name == "segformer" else 128
    # normalize = True # TODO read from config
    save_dir = save_dir / model_name / dataset.name / f"epoch_{epoch}"
    save_dir.mkdir(parents=True, exist_ok=True)

    test_set = LandslideDataset(fold, image_sz=imgsz, mask_sz=imgsz, mean=mean, std=std, do_normalize=True)
    test_loader = dataloader(test_set, batch_size=16, workers=0, shuffle=False, mode="test")
    model = model.to(device)
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs).cpu()
        outputs = F.interpolate(F.sigmoid(logits), size=targets.shape[-2:], mode="bilinear", align_corners=False)
        auroc_metric.update(outputs.flatten(), targets.flatten())
        for confmat in confmats:
            confmat.update(outputs, targets)

        visualise_batch(
            images=inputs.cpu(),         # still normalised
            masks=targets.cpu(),
            preds=outputs.cpu(),
            save_path=save_dir / f"grid_{batch_idx:03d}.png",
            mean=mean,
            std=std,
        )


    
    return {
        "model": model_name,
        "dataset": dataset.name,
        "epoch": epoch,
        "auroc": auroc_metric.compute().item(),
        "TP": confmats[10].tp,
        "TN": confmats[10].tn,
        "FP": confmats[10].fp,
        "FN": confmats[10].fn,
        **{f"F1@{t:.2f}": confmat.f1 for t, confmat in zip(tresholds, confmats)}
    }


def download_run_weights(run, alias: str = "best") -> tuple[Path, int] | None:
    """
    Downloads model artifact from a given run.
    """
    for artifact in run.logged_artifacts():
        if artifact.type == 'model' and alias in artifact.aliases:
            epoch = artifact.version[1:]
            return Path(artifact.download()) / "last.pth", epoch
    raise ValueError(f"No model artifact found with alias '{alias}' in run {run.id}")


def main(root: Path, entity: str, project: str, device: str = "cpu"):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")  # Replace with your entity and project name

    bs = []
    ls = []
    for run in runs:
        model = run.config['model']
        dataset = root.joinpath(run.config['dataset'])
        checkpoint, epoch = download_run_weights(run, alias="best")
        bs.append(run_test(model, dataset, checkpoint, epoch, device))
        checkpoint, epoch = download_run_weights(run, alias="last")
        ls.append(run_test(model, dataset, checkpoint, epoch, device))
    pd.DataFrame(bs).to_csv(root / "best_results.csv", index=False)
    pd.DataFrame(ls).to_csv(root / "last_results.csv", index=False)
    print("Best results saved to best_results.csv")
    


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Run tests on Landslide models.")
    parser.add_argument("--root", type=Path, default=os.getcwd(), help="Root directory for datasets and models.")
    parser.add_argument("--device", type=str, default="mps:0", help="Device to run the model on (e.g., 'cpu', 'cuda', 'mps', etc.).")
    parser.add_argument("--entity", type=str, default="gianluca-calo11", help="WandB entity name.")
    parser.add_argument("--project", type=str, default="landslide-alpha", help="WandB project name.")
    args = parser.parse_args()
    root = Path(args.root)
    main(root, args.entity, args.project, args.device)