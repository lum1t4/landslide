import glob
import os
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Union

import h5py
import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.v2 import functional as F

from landslide.torch_utils import RANK, DistributedEvalSampler, seed_worker
from landslide.utils import yaml_load, yaml_save

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


def img_to_mask(im: Path) -> Path:
    """by default suppose img is under image folder which as sibling mask folder where mask is located"""
    return im.parent.parent.joinpath("mask", im.name.replace("image", "mask"))


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


def preprocess(
    img: Image.Image,
    do_resize: bool = True,
    do_rescale: bool = True,
    do_reduce: bool = False,
    do_normalize: bool = False,
    size: Union[Dict[str, int], List[int], int] = 512,
    resample: int = 2,
    rescale_factor: Optional[float] = 1 / 255,
    image_mean: Optional[Union[float, List[float]]] = [0.485, 0.456, 0.406],
    image_std: Optional[Union[float, List[float]]] = [0.229, 0.224, 0.225],
):
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/segformer/image_processing_segformer.py#L191
    x = F.to_image(img)
    if do_reduce:
        x = reduce_label(x)
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


class SegmentationDataset(Dataset):
    def __init__(self,
        images: List[Path],
        masks: List[Path] = None,
        img_to_mask_fn: Callable = img_to_mask,
        image_sz: int = 128,
        mask_sz: int = 128,
        do_resize: bool = True,
        do_reduce: bool = False,  # only for labels
        do_normalize: bool = False,  # only for imgs
        do_rescale: bool = True,
        mean: list[float] = [0.485, 0.456, 0.406],
        std: list[float] = [0.229, 0.224, 0.225],
    ):
        super().__init__()
        self.images = images
        self.masks = masks
        self.img_to_mask_fn = img_to_mask_fn
        self.image_sz = image_sz
        self.mask_sz = mask_sz
        self.normalize = do_normalize
        self.resize = do_resize
        self.reduce = do_reduce
        self.mean = mean
        self.std = std
        self.rescale = do_rescale
    
    def preprocess_mask(self, mask: Image.Image):
        return preprocess(
            mask,
            do_resize=self.resize,
            size=self.mask_sz,
            do_reduce=self.reduce,
            do_normalize=False,
            do_rescale=True,
        )

    def preprocess_img(self, img: Image.Image):
        return preprocess(
            img,
            do_resize=self.resize,
            size=self.image_sz,
            do_reduce=False,
            image_mean=self.mean,
            image_std=self.std,
            do_rescale=self.rescale,
            do_normalize=self.normalize,
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.images[idx])
        mask_path = self.img_to_mask_fn(self.images[idx])
        if self.masks is not None:
            mask_path = self.masks[idx]
        mask = Image.open(mask_path).convert("L")
        return self.preprocess_img(img), self.preprocess_mask(mask)


class LandslideDataset(SegmentationDataset):
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
    ):
        super().__init__(
            [Path(f) for f in get_images(path)], 
            image_sz=image_sz,
            mask_sz=mask_sz,
            do_resize=do_resize,
            do_reduce=do_reduce,
            do_normalize=do_normalize,
            do_rescale=do_rescale,
            mean=mean,
            std=std
        )


class H5Dataset(Dataset):
    def __init__(self, path: str | Path):
        path = Path(path) if isinstance(path, str) else path
        self.im_files = sorted(path.glob("**/img/*.h5"), key=lambda x: x.stem)
        self.mask_files = [img_to_mask(f) for f in self.im_files]

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, idx):
        with (
            h5py.File(self.im_files[idx], "r") as i,
            h5py.File(self.mask_files[idx], "r") as m,
        ):
            img = i["img"][:]
            mask = m["mask"][:]

        img = np.asarray(img, np.float32).transpose((-1, 0, 1))  # (H, W, C) -> (C, H, W)
        mask = np.asarray(mask, np.float32)
        return img, mask


def dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    workers: int = 8,
    shuffle: bool = True,
    pin_memory: bool = True,
    collate_fn: Optional[Callable] = None,
    mode: Literal["train", "valid", "test"] = "train",
) -> DataLoader:
    bs = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(
        [os.cpu_count() // max(nd, 1), bs if bs > 1 else 0, workers]
    )  # number of workers
    shuffle = shuffle and mode == "train"
    sampler = DistributedSampler if mode == "train" else DistributedEvalSampler
    sampler = None if RANK == -1 else sampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=nw,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def dataset_read_config(descriptor) -> dict:
    assert descriptor.exists(), f"Dataset descriptor not found in {descriptor.parent.as_posix()}"
    content = yaml_load(descriptor)
    path: Path = content.get("dataset", descriptor.parent)
    for fold in ["train", "valid", "test"]:
        content[fold] = path.absolute().joinpath(content.get(fold, fold))
    return content


def dataset_compute_stats(data_root: Path, batch_size: int = 32):
    """
    Compute dataset statistics for training:
      - Number of mask classes
      - Per-channel mean and standard deviation of images
      - Class-weight for positive (landslide) pixels

    Args:
        data_root (Path): Root directory of the dataset (contains train/, valid/, test/).
        batch_size (int): Batch size for the data loader.

    Returns:
        num_classes (int): Number of mask channels (1 for binary, >1 for multi-class).
        mean (Tensor[C]): Mean pixel value per channel.
        std  (Tensor[C]): Standard deviation per channel.
        pos_weight (float): Weight to balance positive vs negative pixels in loss.
    """
    # Prepare the dataset (no normalization yet)
    img_dir = data_root / "train" / "img"
    dataset = LandslideDataset(img_dir, do_normalize=False, image_sz=512)

    # Inspect one sample to get channel & class info
    sample_img, sample_mask = dataset.__getitem__(0)
    num_channels = sample_img.shape[0]
    # If mask is H×W it's binary; else first dimension = number of classes
    num_classes = 1 if sample_mask.ndim == 2 else sample_mask.shape[0]

    # Accumulators for pixel sums
    channel_sum = torch.zeros(num_channels)
    channel_sq_sum = torch.zeros(num_channels)

    # Counters for pixels and patches
    pixel_count = 0               # total number of pixels across all images
    patch_count = 0               # total number of mask patches
    target_count = 0              # total number of pixels across all masks
    pos_patch_count = 0           # number of patches with at least one positive pixel
    pos_target_count = 0          # total number of positive (landslide) pixels
    pos_target_in_pos_patches = 0 # positive pixels only within positive patches

    # Create a loader to iterate over the dataset
    loader = dataloader(
        dataset,
        batch_size=batch_size,
        workers=0,
        shuffle=False,
        mode="valid"
    )

    # Iterate batches and accumulate statistics
    for images, masks in loader:
        # images.shape = [B, C, H, W]
        B, C, H, W = images.shape

        # Sum of pixels and squared pixels per channel
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sq_sum += (images ** 2).sum(dim=[0, 2, 3])

        # Update total pixel count
        pixel_count += B * H * W
        target_count += masks.numel()

        patch_count += B

        # Examine each sample mask in the batch
        for mask in masks:
            # Create boolean mask of positive pixels
            mask = (mask == 1)
            sample_target_pos = mask.sum().item()
            if sample_target_pos > 0:
                pos_patch_count += 1
                pos_target_in_pos_patches += sample_target_pos
                pos_target_count += sample_target_pos

    # Compute mean and standard deviation per channel
    mean = channel_sum / pixel_count
    var = (channel_sq_sum / pixel_count) - mean ** 2
    std = torch.sqrt(var)

    target_weight = (target_count - pos_target_count) / pos_target_count
    # target_inv_freq = target_count / pos_target_count

    return num_classes, mean, std, target_weight



def dataset_write_config(path: str, batch_size: int = 32):
    """
    Compute stats and write a config.yaml in the dataset root.

    Args:
        path (str): Path to dataset root containing train/, valid/, test/.
        batch_size (int): Batch size to use when computing statistics.
        verbose (bool): Whether to log progress and results.
    """
    path = Path(path)
    config = path / "config.yaml"
    data = yaml_load(config) if config.exists() else {}
    nc, mean, std, pos_weights = dataset_compute_stats(path, batch_size=batch_size)
    data["nc"] = nc
    data["name"] = path.name
    data["train"] = "train/img"

    train = path / "train" / "img"
    valid = path / "valid" / "img"
    test = path / "test" / "img"

    data["train"] = train.relative_to(path).as_posix()
    
    if valid.exists():
        data["valid"] = valid.relative_to(path).as_posix()
        if test.exists():
            data["test"] = test.relative_to(path).as_posix()
    elif test.exists():
        data["valid"] = test.relative_to(path).as_posix()

    data["mean"] = mean.tolist()
    data["std"] = std.tolist()
    data["pos_weights"] = pos_weights
    yaml_save(config, data)
