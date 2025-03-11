from pathlib import Path
import h5py
import numpy as np
import cv2
import concurrent.futures
from tqdm import tqdm
import argparse
import os
from functools import partial

# Mapping for renaming directories
rename_map = {
    "TrainData": "train",
    "ValidData": "valid",
    "TestData": "test"
}

def im2mask(im: Path) -> Path:
    return im.parent.parent.joinpath('mask', im.name.replace('image', 'mask'))

def load_image(im_file: Path) -> np.ndarray:
    return cv2.imread(im_file.as_posix(), cv2.IMREAD_UNCHANGED)

def process_file(im_file: Path, src: Path, dst: Path):
    image = load_image(im_file)
    mask = load_image(im2mask(im_file))
    if image.shape[2] == 4:
        image = image[..., :3] # RGBA to RGB

    p = im_file.parent.relative_to(src)
    for key in rename_map:
        p = Path(str(p).replace(key, rename_map[key]))
    p = dst / p
    fname = p / f"{im_file.stem}.png"
    dst_mask_file = im2mask(fname)

    p.mkdir(parents=True, exist_ok=True)
    dst_mask_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(fname.as_posix(), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(dst_mask_file.as_posix(), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def main(src: Path, dst: Path):
    im_files = sorted(src.glob('**/img/*.tif'), key=lambda x: x.name)
    ms_files = [im2mask(im) for im in im_files]
    assert len(im_files) == len(ms_files), "Number of images and masks do not match."

    # Prepare a partial function to include src and dst parameters
    process = partial(process_file, src=src, dst=dst)
    
    # Use ProcessPoolExecutor to parallelize file processing with a progress bar.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # executor.map returns an iterator that we wrap with tqdm for progress tracking.
        list(tqdm(executor.map(process, im_files), total=len(im_files)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default=f'./dataset{os.sep}raw{os.sep}Ischia', help='Source directory')
    parser.add_argument('--dst', type=str, default=f'./dataset{os.sep}processed{os.sep}Ischia', help='Destination directory')
    args = parser.parse_args()
    main(Path(args.src), Path(args.dst))
