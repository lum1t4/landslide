"""
This script takes land-anomalies folder and for each location ("A19_5cm", "Avigliano2", etc.)
creates a new folder with the following structure:

├── train # n - 1 of the locations
│   ├── img
│   └── mask
└── valid # the chosen location
    ├── img
    └── mask

To avoid copying the same files multiple times, the script creates symbolic links to the original files.
The script also creates a config.yaml file with the following structure:

```yaml
name: land-anomalies-A19_5cm
train: train/img
valid: valid/img
test: test/img
nc: 1
mean: [0.5, 0.5, 0.5]
std: [0.5, 0.5, 0.5]
pos_weights: 1.0
```


1. It 
"""
from collections import defaultdict
import math
from pathlib import Path
from typing import List, Optional

from landslide.data import dataset_write_config, get_images, img_to_mask

dataset = "land-anomalies"

SOURCE_DATASET = Path("data/raw/") / dataset
DESTINATION_FOLDER = Path("data/processed/")

imgs = []

imgs = defaultdict(list)
total = 0
for location in SOURCE_DATASET.iterdir():
    imgs[location.name] = list(map(Path, get_images(location / "segmentation_512_512" / "img")))
    total += len(imgs[location.name])



def _symlink_copy(src: List[Path], dst: Path, prefix: Optional[str] = None):
    for img_src in src:
        w_prefix = f"{prefix}-{img_src.name}" if prefix else img_src.name
        img_dst = dst  / w_prefix
        mask_src = img_to_mask(img_src)
        mask_dst = img_to_mask(img_dst)
        mask_dst.parent.mkdir(parents=True, exist_ok=True)
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        
        if img_dst.exists() and img_dst.is_symlink():
            img_dst.unlink()
            mask_dst.unlink()
        img_dst.symlink_to(img_src.absolute())
        mask_dst.symlink_to(mask_src.absolute())


for location in SOURCE_DATASET.iterdir():

    print("Current location", location.name)

    complementary_keys = imgs.keys() - {location.name}
    # copy on train fold the complementary locations
    for key in complementary_keys:
        _symlink_copy(imgs[key], DESTINATION_FOLDER / location.name / "train" / "img", prefix=key)
    
    # copy on validation fold the current location
    _symlink_copy(imgs[location.name], DESTINATION_FOLDER / location.name / "valid" / "img", prefix=location.name)

    dataset_write_config(DESTINATION_FOLDER / location.name)


