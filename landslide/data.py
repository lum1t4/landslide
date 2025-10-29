import glob
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Self, Union
import warnings

import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as F

from landslide.utils import ROOT, yaml_load

IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]


def get_images(path: str | Path, prefix="⚠️"):
    """Read image files."""
    try:
        f = []  # image files
        for p in path if isinstance(path, list) else [path]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                # F = list(p.rglob('*.*'))  # pathlib
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [
                        x.replace("./", parent) if x.startswith("./") else x for x in t
                    ]  # local to global path
                    # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
            else:
                raise FileNotFoundError(f"{prefix}{p} does not exist")
        im_files = sorted(
            x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS
        )
        # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
        assert im_files, f"{prefix}No images found in {path}."
    except Exception as e:
        raise FileNotFoundError(f"{prefix}Error loading data from {path}\n") from e
    return im_files


def img_to_mask(im: Path, mask_dirname: str = "mask") -> Path:
    """by default suppose img is under image folder which as sibling mask folder where mask is located"""
    parts = list(im.parts)
    
    # Start from the second last element and move up (closest to leaf first)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i].lower() in ("image", "img"):
            parts[i] = mask_dirname
            return Path(*parts)
    
    # If no match found, return the path unchanged
    return im


def reduce_label(label: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    For datasets where 0 is used for background, and background itself is
    not included in all classes of a dataset (e.g. ADE20k).
    The background label will be replaced by 255.
    """
    # Avoid using underflow conversion
    label[label == 0] = 255
    label = label - 1
    label[label == 254] = 255
    return label

def preprocess_mask(
    img: Image.Image,
    do_resize: bool = True,
    do_reduce: bool = False,
    size: Union[Dict[str, int], List[int], int] = 512,
    resample: int = 2,
):
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/segformer/image_processing_segformer.py#L191
    x = F.to_image(img)
    if do_reduce:
        x = reduce_label(x)
    if do_resize:
        size = size if not isinstance(size, dict) else (size["height"], size["width"])
        x = F.resize(x, size, interpolation=resample)
    x = F.to_dtype(x, torch.float32, scale=False)
    x = x / 255.0  # Scale from [0, 255] to [0, 1]
    # x = x.squeeze(0)
    return x


def preprocess_img(
    img: Image.Image,
    do_resize: bool = True,
    do_rescale: bool = True,
    do_normalize: bool = False,
    size: Union[Dict[str, int], List[int], int] = 512,
    resample: int = 2,
    rescale_factor: Optional[float] = 1 / 255,
    image_mean: Optional[Union[float, List[float]]] = [0.485, 0.456, 0.406],
    image_std: Optional[Union[float, List[float]]] = [0.229, 0.224, 0.225],
):
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/segformer/image_processing_segformer.py#L191
    x = F.to_image(img)
    if do_resize:
        size = size if not isinstance(size, dict) else (size["height"], size["width"])
        x = F.resize(x, size, interpolation=resample)
    x = F.to_dtype(x, torch.float32, scale=False)
    if do_rescale:
        x = x * rescale_factor
    if do_normalize:
        x = F.normalize(x, mean=image_mean, std=image_std)
    return x


def postprocess(masks, targets, conf: float = 0.5):
    # masks (B, C, H, W), targets (B, H, W)
    size = targets
    if isinstance(targets, torch.Tensor):
        size = targets.shape[-2:]

    if masks.shape[-2:] != size:
        masks = F.interpolate(masks, size=size, mode="bilinear", align_corners=False)

    if masks.shape[1] == 1:
        masks = torch.sigmoid(masks) > conf
    else:
        masks = torch.argmax(masks, dim=1)
    return masks.to(torch.uint8)


class LandslideDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        image_sz: int = 128,
        mask_sz: int = 128,
        do_resize: bool = True,
        do_reduce: bool = False,  # only for labels
        do_normalize: bool = False,  # only for imgs
        do_rescale: bool = True,
        mean: list[float] = [0.485, 0.456, 0.406],
        std: list[float] = [0.229, 0.224, 0.225],
        split: Literal['train', 'valid', 'test'] = "train"
    ):
        self.image_sz = image_sz
        self.mask_sz = mask_sz
        self.normalize = do_normalize
        self.resize = do_resize
        self.reduce = do_reduce
        self.mean = mean
        self.std = std
        self.rescale = do_rescale
        self.split = split

        self.data = yaml_load(path)
        self.root = Path(self.data['root'])
        self.root = self.root if self.root.is_absolute() else ROOT / self.root

        assert self.root.exists(), f"Dataset location could not be found at {self.root}"
        assert Path(path).exists(), f"Index location could not be found at {path}"
        print(f"Loading {self.split} dataset from {path} with root at {self.root}")


        self.images = list(map(lambda x: self.root / x, self.data[self.split]))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index: int) -> dict:
        img_path = self.images[index]
        mask_path = img_to_mask(img_path)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = preprocess_img(
            img,
            do_resize=self.resize,
            size=self.image_sz,
            image_mean=self.mean,
            image_std=self.std,
            do_rescale=self.rescale,
            do_normalize=self.normalize,
        )
        mask = preprocess_mask(
            mask,
            do_resize=self.resize,
            size=self.mask_sz,
            do_reduce=self.reduce,
        )

        return {
            "image_path": img_path,
            "mask_path": mask_path,
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