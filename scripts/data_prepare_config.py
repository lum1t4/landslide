from pathlib import Path
from landslide.dtypes import IterableSimpleNamespace
from landslide.utils import yaml_load, yaml_save
import argparse
from landslide.data import LandslideDataset, dataloader
from pathlib import Path
import torch
import numpy as np
from matplotlib import pyplot as plt


def compute_mean_and_std(path):
    dataset = LandslideDataset(path / 'train' / 'img', do_normalize=False)
    (img, mask) = dataset.__getitem__(0)
    C, H, W = img.shape
    if len(mask.shape) == 2:
        nc = 1
    else:
        nc = mask.shape[0]

    pixel_sum = torch.zeros(C)
    pixel_squared_sum = torch.zeros(C)


    n_patch = len(dataset)
    patch_count_pos = 0
    patch_count_neg = 0

    pixel_count = 0
    pixel_count_pos = 0  # Count of positive (landslide) pixels

    pixel_landslide_patch_count = 0
    pixel_landslide_patch_count_pos = 0

    loader = dataloader(dataset, batch_size=32, workers=0, shuffle=False, mode='valid')

    for images, masks in loader:
        B, _, H, W = images.shape
        # Sum up all pixel values and their squares for each channel
        pixel_sum += images.sum(dim=[0, 2, 3])  # Sum across batch, height, and width
        pixel_squared_sum += (images ** 2).sum(dim=[0, 2, 3])
        # Count total number of pixels
        
        for si in range(B):
            mask = masks[si] == 1
            
            sample_pixel_count = mask.numel()
            sample_pixel_count_pos = mask.sum().item()

            if sample_pixel_count_pos > 0:
                patch_count_pos += 1
                pixel_count_pos += sample_pixel_count_pos
                pixel_landslide_patch_count += sample_pixel_count
                pixel_landslide_patch_count_pos += sample_pixel_count_pos
            else:
                patch_count_neg += 1

            pixel_count += sample_pixel_count
            

    mean = pixel_sum / pixel_count
    var = (pixel_squared_sum / pixel_count) - (mean ** 2) # std = sqrt(E[X^2] - (E[X])^2)
    std = torch.sqrt(var)

    assert n_patch == patch_count_pos + patch_count_neg, "Patch count mismatch"
    
    print(f"On landslide patches, there are {pixel_landslide_patch_count_pos} positive pixels out of {pixel_landslide_patch_count} total pixels. ({100 * pixel_landslide_patch_count_pos / pixel_landslide_patch_count:.1f}%)")


    pixel_count_neg = pixel_count - pixel_count_pos
    pixel_pos_freq = pixel_count_pos / pixel_count
    pixel_pos_inv_freq = pixel_count / pixel_count_pos
    pixel_pos_weight = pixel_count_neg / pixel_count_pos


    print("\n" + "="*50)
    print("Pixel Statistics")
    print("="*50)
    print(f"pixel_count: {pixel_count}")
    print(f"pixel_count_pos: {pixel_count_pos}")
    print(f"pixel_count_neg: {pixel_count_neg}")

    print(f"pixel_pos_weight: {pixel_pos_weight:.4f}")
    print(f"pixel_pos_freq: {pixel_pos_freq:.4f}")
    print(f"pixel_pos_inv_freq: {pixel_pos_inv_freq:.4f}")
    print(f"pixel_pos_weight (from pos/neg): {pixel_pos_weight:.4f}")
    print("="*50 + "\n")


    print("\n" + "="*50)
    print("Dataset Statistics")
    print("="*50)
    print(f"pos_weight: {pixel_pos_weight:.4f}")
    print(f"mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"std_dev:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    print(f"num landslide patches: {patch_count_pos}/{n_patch} ({100 * patch_count_pos / n_patch:.1f}%)")
    print("="*50 + "\n")
    return nc, mean, std, pixel_pos_weight


def main(path: str, batch_size: int = 32):
    path = Path(path)
    config = path / "config.yaml"
    if config.exists():
        data = yaml_load(config)
    else:
        data = {}
    
    nc, mean, std, pos_weights = compute_mean_and_std(path)
    data['nc'] = nc
    data["train"] = "train/img"

    valid = path / "valid" / "img"
    test = path / "test" / "img"

    if valid.exists():
        data["valid"] = valid.relative_to(path).as_posix()
        if test.exists():
            data["test"] = test.relative_to(path).as_posix()
    elif test.exists():
        data["valid"] = test.relative_to(path).as_posix()
        if "test" in data:
            data.pop("test")

    data["mean"] = mean.tolist()
    data["std"] = std.tolist()
    data["pos_weights"] = pos_weights
    yaml_save(config, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for landslide detection model training.")
    parser.add_argument("--dataset", type=str, default="dataset/processed/A19", help="Path to the dataset YAML file.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for data loader.")

    args = parser.parse_args()
    main(Path(args.dataset), batch_size=args.batch)