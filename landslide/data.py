
from functools import partial
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import h5py
import os
import torch
import numpy as np
from typing import Callable, Optional, Literal, Union, List, Dict
import glob
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
import PIL.Image as Image
from landslide.torch import DistributedEvalSampler, seed_worker, RANK, LOCAL_RANK
from landslide.utils import yaml_load

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


def im2mask(im: Path) -> Path:
    return im.parent.parent.joinpath('mask', im.name.replace('image', 'mask'))


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
        image_std: Optional[Union[float, List[float]]] = [0.229, 0.224, 0.225]
):
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/segformer/image_processing_segformer.py#L191
    x = F.to_image(img)
    if do_reduce:
        x = reduce_label(x)
    if do_resize:
        size = size if not isinstance(size, dict) else (size['height'], size['width'])
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
            do_reduce: bool = False, # only for labels
            do_normalize: bool = False, # only for imgs
            do_rescale: bool = True,
            mean: list[float] = [0.485, 0.456, 0.406],
            std: list[float] = [0.229, 0.224, 0.225],
    ):
        self.files = [Path(f) for f in get_images(path)]
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
            do_rescale=self.rescale,
            do_normalize=self.normalize
        )

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        mask = Image.open(im2mask(self.files[idx]))
        return self.preprocess_img(img), self.preprocess_mask(mask)
    

class H5Dataset(Dataset):
    def __init__(self, path: str | Path):
        path = Path(path) if isinstance(path, str) else path
        self.im_files = sorted(path.glob('**/img/*.h5'), key=lambda x: x.stem)
        self.mask_files = [im2mask(f) for f in self.im_files]
    
    def __len__(self):
        return len(self.im_files)
    
    def __getitem__(self, idx):
        with h5py.File(self.im_files[idx], 'r') as i, h5py.File(self.mask_files[idx], 'r') as m:
            img = i['img'][:]
            mask = m['mask'][:]

        img = np.asarray(img, np.float32).transpose((-1, 0, 1))  # (H, W, C) -> (C, H, W)
        mask = np.asarray(mask, np.float32)
        return img, mask


# LS3 = partial(LandslideDataset,
#     num_channels=3,
#     mean=[-0.3074, -0.1277, -0.0625],
#     std=[0.8775, 0.8860, 0.8869]
# )
# 
# LS14 = partial(
#     LandslideDataset,
#     ch=14,
#     mean=[-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819],
#     std=[0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]
# )


def dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    workers: int = 8,
    shuffle: bool = False,
    pin_memory: bool = True,
    collate_fn: Optional[Callable] = None,
    mode: Literal['train', 'valid', 'test'] = "train",
) -> DataLoader:
    bs = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), bs if bs > 1 else 0, workers])  # number of workers
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


def load_dataset(name: str, root: Path = Path('./dataset/processed/')):
    descriptor = root / name / 'config.yaml'
    assert descriptor.exists(), f"Dataset descriptor not found in {descriptor.parent.as_posix()}"
    content = yaml_load(descriptor)
    path: Path = content.get("dataset", descriptor.parent)
    for fold in ["train", "valid", "test"]:
        content[fold] = path.absolute().joinpath(content.get(fold, fold))
    return content


