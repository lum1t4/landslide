import glob
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Self, Union

import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as F

from landslide.torch_utils import dataloader

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
        index_path: str | Path,
        dataset_path: str | Path,
        image_sz: int = 128,
        mask_sz: int = 128,
        do_resize: bool = True,
        do_reduce: bool = False,  # only for labels
        do_normalize: bool = False,  # only for imgs
        do_rescale: bool = True,
        mean: list[float] = [0.485, 0.456, 0.406],
        std: list[float] = [0.229, 0.224, 0.225],
        split: str = "train"
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

        dataset_path = Path(dataset_path)
        index_path = Path(index_path)  / f'{self.split}.csv'

        assert dataset_path.exists(), f"Dataset location could not be found at {dataset_path}"
        assert index_path.exists(), f"Index location could not be found at {index_path}"

        df = pd.read_csv(index_path, names=["image"])
        images = map(lambda x: x.replace("/", os.sep), df["image"])
        images = map(lambda x: dataset_path.joinpath(x), images)
        images = filter(lambda x: x.exists(), images)
        self.images = list(images)

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
    
    def compute_stats(self) -> Self:
        if self.split != "train":
            warnings.warn("You should not use any fold beside train to compute dataset stats")

        # Deactivate normalization and reinstate it later
        current = self.normalize
        self.normalize = False
        loader = dataloader(self, batch_size=12, mode="valid")
        item = self.__getitem__(0)
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

        self.mean = mean.tolist()
        self.std = std.tolist()
        self.pixel_count = pixel_count
        self.pixel_positive = pixel_positive
        self.pixel_weight = pixel_weight
        self.patch_count = patch_count
        self.patch_positive = patch_positive
        self.patch_weight = patch_weight
        self.normalize = current
        print("Computed stats")
        return self
