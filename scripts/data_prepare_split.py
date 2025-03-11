from pathlib import Path
import h5py
import numpy as np
import cv2
import concurrent.futures
from tqdm import tqdm
import argparse
import os
from functools import partial
import shutil

def im2mask(im: Path) -> Path:
    return im.parent.parent.joinpath('mask', im.name.replace('image', 'mask'))


def process_file(item: tuple[Path, str], src: Path, dst: Path):
    im_file, fold = item

    dest_im_file = dst / fold / 'img' / im_file.name
    dest_im_file.parent.mkdir(parents=True, exist_ok=True)
    dst_mask_file = im2mask(dest_im_file)
    dst_mask_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(im_file, dest_im_file)
    shutil.copy(im2mask(im_file), dst_mask_file)


def main(src: Path, dest: Path):
    im_files = list(src.glob('**/img/*.tif'))
    # split the image files into train, valid sets
    train_files = im_files[:int(0.8 * len(im_files))]
    valid_files = im_files[int(0.8 * len(im_files)):]
    assert len(im_files) == len(train_files) + len(valid_files), "Number of images do not match."

    im_files = [(im_file, 'train') for im_file in train_files] + [(im_file, 'valid') for im_file in valid_files]
    # Prepare a partial function to include src and dst parameters
    process = partial(process_file, src=src, dst=dest)
    
    # Use ProcessPoolExecutor to parallelize file processing with a progress bar.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # executor.map returns an iterator that we wrap with tqdm for progress tracking.
        list(tqdm(executor.map(process, im_files), total=len(im_files)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default=f'./dataset{os.sep}raw{os.sep}A19', help='Source directory')
    parser.add_argument('--dst', type=str, default=f'./dataset{os.sep}interim{os.sep}A19', help='Destination directory')
    args = parser.parse_args()
    main(Path(args.src), Path(args.dst))
