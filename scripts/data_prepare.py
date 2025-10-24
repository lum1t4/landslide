import glob
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Self, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as F

from landslide.data import img_to_mask, preprocess_img, preprocess_mask
from landslide.torch_utils import dataloader
from landslide.utils import yaml_save

root = Path('data/raw/land-anomalies')


class CSVLandslideDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        dataset_path: str | Path,
        image_sz: int = 128,
        mask_sz: int = 128,
        do_resize: bool = True,
        do_reduce: bool = False,  # only for labels
        do_rescale: bool = True,
    ):
        
        self.image_sz = image_sz
        self.mask_sz = mask_sz
        self.resize = do_resize
        self.reduce = do_reduce
        self.rescale = do_rescale

        assert root.exists(), f"Dataset location could not be found at {dataset_path}"
        assert path.exists(), f"Index location could not be found at {path}"

        df = pd.read_csv(path, names=["image"])
        images = map(lambda x: x.replace("/", os.sep), df["image"])
        images = map(lambda x: root.joinpath(x), images)
        images = filter(lambda x: x.exists(), images)
        self.images = list(images)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index: int) -> dict:
        path_img = self.images[index]
        path_msk = img_to_mask(path_img)

        img = Image.open(path_img).convert("RGB")
        mask = Image.open(path_msk).convert("L")

        img = preprocess_img(
            img, do_resize=self.resize,
            size=self.image_sz,
            do_rescale=self.rescale,
            do_normalize=False,
        )
        mask = preprocess_mask(
            mask,
            do_resize=self.resize,
            size=self.mask_sz,
            do_reduce=self.reduce,
        )

        return {
            "image_path": path_img,
            "mask_path": path_msk,
            "input": img,
            "target": mask
        }
    
    @staticmethod
    def collate_fn(items: list[dict]) -> dict:
        """Collate function for PyTorch DataLoader."""
        # Stack tensors into batch
        inputs = torch.stack([item["input"] for item in items])
        targets = torch.stack([item["target"] for item in items])

        # Collect metadata as lists
        image_paths = [item["image_path"] for item in items]
        mask_paths = [item["mask_path"] for item in items]

        return {
            "input": inputs,
            "target": targets,
            "image_path": image_paths,
            "mask_path": mask_paths,
        }


def compute_stats(dataset : CSVLandslideDataset) -> dict:
    loader = dataloader(dataset, batch_size=12, mode="valid")
    item = dataset.__getitem__(0)
    img, mask = item['input'], item['target']

    num_chs = img.shape[0]
    num_cls = 1 if mask.ndim == 2 else mask.shape[0]  # noqa: F841

    ch_sum = torch.zeros(num_chs)
    ch_sqr = torch.zeros(num_chs)

    pixel_count = 0 # How many RGB pixels
    patch_count = 0 # How many images (should be equal to len(dataset))

    pixel_positive = 0 # How many pixel are positive
        # How many patches are positive (have at least one positive pixel) 
    patch_positive = 0

    for batch in loader:
        images = batch['input']
        masks = batch['target']
        B, _, H, W = images.shape
        ch_sum += images.sum(dim=[0, 2, 3])
        ch_sqr += (images ** 2).sum(dim=[0, 2, 3])
        pixel_count += B * H * W
        patch_count += B
        for mask in masks:
            positive = (mask == 1).sum().item()
            if positive > 0:
                patch_positive += 1
                pixel_positive += positive


    mean = ch_sum / pixel_count
    std = torch.sqrt((ch_sqr / pixel_count) - mean ** 2)
    patch_weight = (patch_count - patch_positive) / patch_count
    pixel_weight = (pixel_count - pixel_positive) / pixel_count
    
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "classes": ["landslide"],
        "pixel_count": pixel_count,
        "pixel_positive": pixel_positive,
        "pixel_weight": pixel_weight,
        "patch_count": patch_count,
        "patch_positive": patch_positive,
        "patch_weight": patch_weight,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="Prepare dataset descriptors")
    parser.add_argument("index", type=str, default='data/interim/csv_split/')
    args = parser.parse_args()

    indices = Path(args.index)
    folds = [fold for fold in indices.iterdir() if fold.is_dir()]

    for fold in folds:
        dataset = CSVLandslideDataset(fold / "train.csv", root)
        data = compute_stats(dataset)

        data['root'] = root.as_posix()
        data["train"] = list(map(lambda x: x.relative_to(root).as_posix(), dataset.images))
        dataset = CSVLandslideDataset(fold / "val.csv", root)
        data["valid"] = list(map(lambda x: x.relative_to(root).as_posix(), dataset.images))
        dataset = CSVLandslideDataset(fold / "test.csv", root)
        data["test"] = list(map(lambda x: x.relative_to(root).as_posix(), dataset.images))

        dst = Path('data/processed') / indices.name
        dst.mkdir(parents=True, exist_ok=True)
        yaml_save(dst / f"{fold.name}.yml", data)

