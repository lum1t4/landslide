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
cast = {
    "TrainData": "train",
    "ValidData": "valid",
    "TestData": "test"
}

def im2mask(im: Path) -> Path:
    return im.parent.parent.joinpath('mask', im.name.replace('image', 'mask'))

def load_image(im_file: Path) -> np.ndarray:
    with h5py.File(im_file, 'r') as h:
        assert 'img' in h, f'img not found in {im_file}'
        image = h['img'][:]
    return image

def load_mask(mask_file: Path) -> np.ndarray:
    with h5py.File(mask_file, 'r') as h:
        assert 'mask' in h, f'mask not found in {mask_file}'
        mask = h['mask'][:]
    return mask

def process_file(im_file: Path, src: Path, dst: Path):
    # Load and process the image
    img = load_image(im_file)
    img = img[:, :, [3, 2, 1]]  # Select channels in order: [R, G, B]
    for c in range(3):
        ch_min = img[..., c].min()
        ch_max = img[..., c].max()
        img[..., c] = (img[..., c] - ch_min) / (ch_max - ch_min)
    img = (img * 255).astype(np.uint8)

    # Determine output path for the image file
    p = im_file.parent.relative_to(src)
    for key in cast:
        p = Path(str(p).replace(key, cast[key]))
    p = dst / p

    
    fname = p / f"{im_file.stem}.png"

    # Process the mask
    mask_file = im2mask(im_file)
    mask = load_mask(mask_file)
    mask = (mask * 255).astype(np.uint8)
    
    dst_mask = im2mask(fname)
    dst_mask.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(dst_mask.absolute().as_posix(), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    # Save the processed image
    p.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(fname.absolute().as_posix(), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

def main(src: Path, dst: Path):
    im_files = sorted(src.glob('**/img/*.h5'), key=lambda x: x.name)

    # Prepare a partial function to include src and dst parameters
    process = partial(process_file, src=src, dst=dst)
    
    # Use ProcessPoolExecutor to parallelize file processing with a progress bar.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # executor.map returns an iterator that we wrap with tqdm for progress tracking.
        list(tqdm(executor.map(process, im_files), total=len(im_files)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default=f'./dataset{os.sep}raw{os.sep}L4S', help='Source directory')
    parser.add_argument('--dst', type=str, default=f'./dataset{os.sep}processed{os.sep}L4S', help='Destination directory')
    args = parser.parse_args()
    main(Path(args.src), Path(args.dst))
