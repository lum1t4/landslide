import argparse
from collections import defaultdict
import concurrent.futures
import os
from pathlib import Path
import random
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
        futures = [executor.submit(_symlink_copy_image, img_src, dst, prefix) for img_src in src]
        concurrent.futures.wait(futures)


def main(source: str, destination: str, valid: int = 2, seed: int = 0):
    """
    This script takes main dataset folder (land-anomalies) which has more or less the following structure:

    ```
    {dataset}
    │   {location (e.g. "A19_5cm", "Avigliano2", etc.)}
    │   ├── segmentation_512_512
    │   │   ├── img
    │   │   └── mask
    ...
    ```
    and creates a new folder with the following structure:
    ```
    {dataset}
    ├── {location}
    │   ├── train # data from remainig (n - k - 1) locations 
    │   │   ├── img
    │   │   └── mask
    │   └── valid # data from k randomly selected locations
    │   │   ├── img
    │   │   └── mask
    │   └── test # data from the current location
    │   │   ├── img
    │   │   └── mask
    │   └───config.yaml
    ...
    ```
    To avoid duplicating files, the script creates symbolic links for images and masks.
    The script also creates a config.yaml for each location in the destination folder so metrics like
    mean, std, etc are computed beforhand:
    Example of config.yaml:
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
    # Set random seed for reproducibility of validation selection
    random.seed(seed)

    dataset_src = Path(source)  # ex. data/raw/land-anomalies
    dataset_dst = Path(destination)  # ex. data/processed/land-anomalies
    dataset_dst.mkdir(parents=True, exist_ok=True)

    assert dataset_src.exists(), f"Source dataset {dataset_src} does not exist"
    assert dataset_dst.parent.is_dir(), (
        f"Parent of destination folder {dataset_dst.parent} does not exist or is not a directory"
    )

    imgs = defaultdict(list)

    locations = [location for location in dataset_src.iterdir() if location.is_dir()]

    for location in locations:
        if not location.is_dir():
            continue
        imgs[location.name] = list(
            map(Path, get_images(location / "segmentation_512_512" / "img"))
        )

    for location in locations:

        location_dst = dataset_dst / location.name
        print("Current location", location.name)
        complementary_keys = set(imgs.keys()) - {location.name}

        # Ensure the requested number of validation locations is valid
        if valid > len(complementary_keys):
            raise ValueError(
                f"Number of validation locations ({valid}) is greater than available locations ({len(complementary_keys)})"
            )

        # Randomly select validation locations
        validation_keys = random.sample(sorted(complementary_keys), valid) if valid > 0 else {}
        # Remaining locations are for training
        training_keys = sorted(complementary_keys - set(validation_keys))

        # Symlink training images
        for key in training_keys:
            _symlink_copy(imgs[key], location_dst / "train" / "img", prefix=key)

        # Symlink validation images
        for key in validation_keys:
            _symlink_copy(imgs[key], location_dst / "valid" / "img", prefix=key)

        # Symlink test images (current location)
        _symlink_copy(imgs[location.name], location_dst / "test" / "img")

        # Create config.yaml for the current location
        dataset_write_config(location_dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dataset for land-anomalies")
    parser.add_argument("--source", type=str, help="Path to the dataset folder", required=True)
    parser.add_argument("--destination", type=str, help="The destination folder for the dataset")
    parser.add_argument("--valid", type=int, default=0, help="Number of locations to use for validation set")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for selecting validation locations")
    args = parser.parse_args()
    main(args.source, args.destination, args.valid, args.seed)
