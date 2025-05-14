
import argparse
from collections import defaultdict
import concurrent.futures
import os
from pathlib import Path
from typing import List, Optional

from landslide.data import dataset_write_config, get_images, img_to_mask


def _symlink_copy_image(img_src: Path, dst: Path, prefix: Optional[str] = None):
    """
    Create symbolic links for an image and its corresponding mask.

    Args:
        img_src (Path): Source image file path.
        dst (Path): Destination folder where symlinks will be created.
        prefix (Optional[str]): Optional prefix for the symlink filename.
    """
    w_prefix = f"{prefix}-{img_src.name}" if prefix else img_src.name
    img_dst = dst / w_prefix
    mask_src = img_to_mask(img_src)
    mask_dst = img_to_mask(img_dst)

    # Ensure parent directories exist
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    mask_dst.parent.mkdir(parents=True, exist_ok=True)

    # Unlink if a symlink already exists
    if img_dst.exists() and img_dst.is_symlink():
        img_dst.unlink()
    if mask_dst.exists() and mask_dst.is_symlink():
        mask_dst.unlink()

    # Create symbolic links
    img_dst.symlink_to(img_src.absolute())
    mask_dst.symlink_to(mask_src.absolute())


def _symlink_copy(src: List[Path], dst: Path, prefix: Optional[str] = None):
    """
    Concurrently create symbolic links for a list of image paths and their corresponding masks.

    Args:
        src (List[Path]): List of source image file paths.
        dst (Path): Destination folder to place symbolic links.
        prefix (Optional[str]): Optional prefix to add to the symlink filenames.
    """
    # Use ThreadPoolExecutor for I/O-bound tasks like file operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(_symlink_copy_image, img_src, dst, prefix)
            for img_src in src
        ]
        concurrent.futures.wait(futures)


def main(dataset_src: str, destination: str):
    """
    This script takes main dataset folder (land-anomalies) and for each location ("A19_5cm", "Avigliano2", etc.)
    creates a new folder with the following structure:
    {dataset}
    │   {location}
    │   ├── train # n - 1 of the locations
    │   │   ├── img
    │   │   └── mask
    │   └── valid # the chosen location
    │       ├── img
    │       └── mask
    ...

    To avoid copying the same files multiple times, the script creates symbolic links to the original files.
    The script also creates a config.yaml file with the following structure:

    ```yaml
    name: {location_name}
    train: train/img
    valid: valid/img
    nc: 1
    mean: [0.5, 0.5, 0.5]
    std: [0.5, 0.5, 0.5]
    pos_weights: 1.0
    ```
    """

    dataset = "land-anomalies"

    dataset_src = Path(dataset_src)
    dataset_dst = Path(destination)

    assert dataset_src.exists(), f"Source dataset {dataset_src} does not exist"
    assert dataset_dst.is_dir(), f"Destination folder {dataset_dst} does not exist or is not a directory"

    dataset = dataset_src.name
    dataset_dst = dataset_dst / dataset

    imgs = defaultdict(list)
    for location in dataset_src.iterdir():
        imgs[location.name] = list(map(Path, get_images(location / "segmentation_512_512" / "img")))

    for location in dataset_src.iterdir():
        location_dst = dataset_dst / location.name
        print("Current location", location.name)
        complementary_keys = imgs.keys() - {location.name} # n - 1 of the locations
        # copy on train fold the complementary locations
        for key in complementary_keys:
            _symlink_copy(imgs[key], location_dst / "train" / "img", prefix=key)
        
        # copy on validation fold the current location
        _symlink_copy(imgs[location.name], location_dst / "valid" / "img", prefix=location.name)
        dataset_write_config(location_dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dataset for land-anomalies")
    parser.add_argument("--dataset", type=str, help="The name of the dataset to create")
    parser.add_argument("--destination", type=str, help="The destination folder for the dataset")
    args = parser.parse_args()
    main(args.dataset, args.destination)