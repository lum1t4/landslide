import random
from typing import Callable, Dict, Optional

import cv2
import numpy as np
import PIL.Image as Image


# Helper functions
def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array."""
    return np.array(img)


def numpy_to_pil(arr: np.ndarray, mode: str = "RGB") -> Image.Image:
    """Convert numpy array to PIL Image."""
    return Image.fromarray(arr.astype(np.uint8), mode=mode)


class BaseAugmentation:
    """Base class for augmentations."""

    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of applying the augmentation.
        """
        self.p = p

    def __call__(self, data: Dict) -> Dict:
        """Apply augmentation to data dict containing image and mask."""
        if random.random() < self.p:
            return self.apply(data)
        return data

    def apply(self, data: Dict) -> Dict:
        """Implement augmentation logic. Must return dict with same structure."""
        raise NotImplementedError


class Compose:
    """Compose multiple augmentations."""

    def __init__(self, transforms: list):
        """
        Args:
            transforms: List of augmentation transforms.
        """
        self.transforms = transforms

    def __call__(self, data: Dict) -> Dict:
        """Apply all transforms sequentially."""
        for transform in self.transforms:
            data = transform(data)
        return data

    def insert(self, index: int, transform: Callable) -> None:
        """Insert a transform at a specific position in the pipeline."""
        self.transforms.insert(index, transform)

    def append(self, transform: Callable) -> None:
        """Append a transform to the end of the pipeline."""
        self.transforms.append(transform)


# Augmentation Classes

class RandomFlip(BaseAugmentation):
    """Random flip augmentation."""

    def __init__(self, direction: str = "horizontal", p: float = 0.5):
        """
        Args:
            direction: 'horizontal' or 'vertical'
            p: Probability of applying the augmentation
        """
        super().__init__(p)
        self.direction = direction

    def apply(self, data: Dict) -> Dict:
        """Apply flip to image and mask."""
        img = data["input"]
        mask = data["target"]

        if self.direction == "horizontal":
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        elif self.direction == "vertical":
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
        data["input"] = img
        data["target"] = mask
        return data


class RandomHSV(BaseAugmentation):
    """Random HSV color augmentation (only affects image, not mask)."""

    def __init__(self, hgain: float = 0.5, sgain: float = 0.5, vgain: float = 0.5, p: float = 1.0):
        """
        Args:
            hgain: Hue gain (fraction of 180 degrees)
            sgain: Saturation gain (multiplier)
            vgain: Value gain (multiplier)
            p: Probability of applying the augmentation
        """
        super().__init__(p)
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def apply(self, data: Dict) -> Dict:
        """Apply HSV augmentation to image only."""
        img = data["input"]
        # Random gains
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1

        # Convert to HSV
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))

        # Apply gains with LUT
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

        hue = cv2.LUT(hue, lut_hue)
        sat = cv2.LUT(sat, lut_sat)
        val = cv2.LUT(val, lut_val)

        # Convert back to RGB
        img_hsv = cv2.merge((hue, sat, val))
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        data["input"] = img
        return data


class RandomPerspective(BaseAugmentation):
    """Random perspective and affine transformations."""

    def __init__(
        self,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        border: tuple = (0, 0),
        p: float = 1.0,
    ):
        """
        Args:
            degrees: Rotation range in degrees
            translate: Translation fraction of image size
            scale: Scaling range (e.g., 0.5 means 0.5x to 1.5x)
            shear: Shear range in degrees
            perspective: Perspective distortion range
            border: Border to add (height, width)
            p: Probability of applying the augmentation
        """
        super().__init__(p)
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border

    def apply(self, data: Dict) -> Dict:
        """Apply perspective transform to image and mask."""
        img, mask = data["input"], data["target"]

        height, width = img.shape[:2]

        # Center
        C = np.eye(3)
        C[0, 2] = -width / 2
        C[1, 2] = -height / 2

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = np.tan(random.uniform(-self.shear, self.shear) * np.pi / 180)
        S[1, 0] = np.tan(random.uniform(-self.shear, self.shear) * np.pi / 180)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * height

        # Combined transformation matrix
        M = T @ S @ R @ P @ C

        # Apply transformation
        if (self.border[0] != 0) or (self.border[1] != 0) or (M != np.eye(3)).any():
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
                mask = cv2.warpPerspective(mask, M, dsize=(width, height), borderValue=0)
            else:
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
                mask = cv2.warpAffine(mask, M[:2], dsize=(width, height), borderValue=0)

        data["input"] = img
        data["target"] = mask
        return data


class Mosaic(BaseAugmentation):
    """Mosaic augmentation - combines 4 images into one."""

    def __init__(self, dataset, imgsz: int = 640, p: float = 1.0, n: int = 4):
        """
        Args:
            dataset: Dataset to sample from
            imgsz: Target image size
            p: Probability of applying the augmentation
            n: Number of images to combine (default 4 for 2x2 grid)
        """
        super().__init__(p)
        self.dataset = dataset
        self.imgsz = imgsz if isinstance(imgsz, int) else imgsz[0]
        self.n = n
        self.border = (-self.imgsz // 2, -self.imgsz // 2)

    def apply(self, data: Dict) -> Dict:
        """Apply mosaic augmentation."""
        # Sample 3 additional images
        indices = random.choices(range(len(self.dataset)), k=3)
        mosaic_data = [data] + [self.dataset.__load__(i) for i in indices]

        # Create output arrays
        img4 = np.full((self.imgsz * 2, self.imgsz * 2, 3), 114, dtype=np.uint8)
        mask4 = np.zeros((self.imgsz * 2, self.imgsz * 2), dtype=np.uint8)

        # Random center point
        yc = int(random.uniform(0.5 * self.imgsz, 1.5 * self.imgsz))
        xc = int(random.uniform(0.5 * self.imgsz, 1.5 * self.imgsz))

        for i, d in enumerate(mosaic_data):
            img = d["input"]
            mask = d["target"]
            h, w = img.shape[:2]

            # Resize to target size
            img = cv2.resize(img, (self.imgsz, self.imgsz))
            mask = cv2.resize(mask, (self.imgsz, self.imgsz))

            # Place in grid (top-left, top-right, bottom-left, bottom-right)
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - self.imgsz, 0), max(yc - self.imgsz, 0), xc, yc
                x1b, y1b, x2b, y2b = self.imgsz - (x2a - x1a), self.imgsz - (y2a - y1a), self.imgsz, self.imgsz
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - self.imgsz, 0), min(xc + self.imgsz, self.imgsz * 2), yc
                x1b, y1b, x2b, y2b = 0, self.imgsz - (y2a - y1a), min(self.imgsz, x2a - x1a), self.imgsz
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - self.imgsz, 0), yc, xc, min(self.imgsz * 2, yc + self.imgsz)
                x1b, y1b, x2b, y2b = self.imgsz - (x2a - x1a), 0, self.imgsz, min(y2a - y1a, self.imgsz)
            elif i == 3:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + self.imgsz, self.imgsz * 2), min(self.imgsz * 2, yc + self.imgsz)
                x1b, y1b, x2b, y2b = 0, 0, min(self.imgsz, x2a - x1a), min(y2a - y1a, self.imgsz)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            mask4[y1a:y2a, x1a:x2a] = mask[y1b:y2b, x1b:x2b]

        # Crop to final size
        img4 = img4[self.imgsz // 2 : self.imgsz // 2 + self.imgsz, self.imgsz // 2 : self.imgsz // 2 + self.imgsz]
        mask4 = mask4[self.imgsz // 2 : self.imgsz // 2 + self.imgsz, self.imgsz // 2 : self.imgsz // 2 + self.imgsz]

        data["input"] = img4
        data["target"] = mask4
        return data


class MixUp(BaseAugmentation):
    """MixUp augmentation - blends two images."""

    def __init__(self, dataset, pre_transform=None, p: float = 0.0):
        """
        Args:
            dataset: Dataset to sample from
            pre_transform: Transform to apply before mixing
            p: Probability of applying the augmentation
        """
        super().__init__(p)
        self.dataset = dataset
        self.pre_transform = pre_transform

    def apply(self, data: Dict) -> Dict:
        """Apply mixup augmentation."""
        # Sample second image
        idx = random.randint(0, len(self.dataset) - 1)
        data2 = self.dataset.__load__(idx)

        if self.pre_transform:
            data2 = self.pre_transform(data2)

        img1 = data["input"]
        mask1 = data["target"]
        img2 = data2["input"]
        mask2 = data2["target"]

        # Ensure same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]))

        # Beta distribution for mixing ratio
        r = np.random.beta(32.0, 32.0)

        # Blend
        img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
        mask = (mask1 * r + mask2 * (1 - r)).astype(np.uint8)

        data["input"] = img
        data["target"] = mask
        return data


class CutMix(BaseAugmentation):
    """CutMix augmentation - cuts and pastes rectangular regions."""

    def __init__(self, dataset, pre_transform=None, p: float = 0.0):
        """
        Args:
            dataset: Dataset to sample from
            pre_transform: Transform to apply before mixing
            p: Probability of applying the augmentation
        """
        super().__init__(p)
        self.dataset = dataset
        self.pre_transform = pre_transform

    def apply(self, data: Dict) -> Dict:
        """Apply cutmix augmentation."""
        # Sample second image
        idx = random.randint(0, len(self.dataset) - 1)
        data2 = self.dataset.__load__(idx)

        if self.pre_transform:
            data2 = self.pre_transform(data2)

        img1, mask1 = data["input"], data["target"]
        img2, mask2 = data2["input"], data2["target"]

        # Ensure same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]))

        h, w = img1.shape[:2]

        # Random box size from beta distribution
        cut_ratio = np.random.beta(1.0, 1.0)
        cut_w = int(w * np.sqrt(cut_ratio))
        cut_h = int(h * np.sqrt(cut_ratio))

        # Random box position
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)

        # Cut and paste
        img1[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
        mask1[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]

        data["input"] = img1
        data["target"] = mask1
        return data


class CopyPaste(BaseAugmentation):
    """CopyPaste augmentation - overlays images with mask-based blending."""

    def __init__(self, dataset=None, pre_transform=None, p: float = 0.0, mode: str = "flip"):
        """
        Args:
            dataset: Dataset to sample from (required for 'mixup' mode)
            pre_transform: Transform to apply before mixing
            p: Probability of applying the augmentation
            mode: 'flip' or 'mixup'
        """
        super().__init__(p)
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.mode = mode

    def apply(self, data: Dict) -> Dict:
        """Apply copy-paste augmentation."""
        img, mask = data["input"], data["target"]

        if self.mode == "flip":
            # Simple flip mode: overlay flipped version
            img_flip = np.fliplr(img).copy()
            mask_flip = np.fliplr(mask).copy()

            # Use mask as alpha for blending
            alpha = (mask_flip > 0).astype(np.float32)
            if len(alpha.shape) == 2:
                alpha = alpha[:, :, np.newaxis]

            img = (img * (1 - alpha) + img_flip * alpha).astype(np.uint8)
            mask = np.maximum(mask, mask_flip)

        elif self.mode == "mixup" and self.dataset:
            # Sample another image
            idx = random.randint(0, len(self.dataset) - 1)
            data2 = self.dataset.__load__(idx)

            if self.pre_transform:
                data2 = self.pre_transform(data2)

            img2, mask2 = data2["input"], data2["target"]

            # Ensure same size
            if img.shape != img2.shape:
                img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
                mask2 = cv2.resize(mask2, (mask.shape[1], mask.shape[0]))

            # Use mask2 as alpha for blending
            alpha = (mask2 > 0).astype(np.float32)
            if len(alpha.shape) == 2:
                alpha = alpha[:, :, np.newaxis]

            img = (img * (1 - alpha) + img2 * alpha).astype(np.uint8)
            mask = np.maximum(mask, mask2)

        data["input"] = img
        data["target"] = mask
        return data


class Albumentations(BaseAugmentation):
    """Wrapper for Albumentations library."""

    def __init__(self, p: float = 1.0, transforms=None):
        """
        Args:
            p: Probability of applying the augmentation
            transforms: List of albumentations transforms config
        """
        super().__init__(p)
        self.transform = None

        # Try to import albumentations
        try:
            import albumentations as A

            if transforms is not None:
                # Build albumentations pipeline from config if provided
                self.transform = A.Compose([A.Blur(p=0.1), A.MedianBlur(p=0.1), A.ToGray(p=0.01)])
            else:
                # Default augmentations
                self.transform = A.Compose([A.Blur(p=0.1), A.MedianBlur(p=0.1), A.ToGray(p=0.01)])

        except ImportError:
            pass  # Albumentations not installed, skip

    def apply(self, data: Dict) -> Dict:
        """Apply albumentations to image and mask."""
        if self.transform is None:
            return data

        img = data["input"]
        mask = data["target"]

        # Apply albumentations
        transformed = self.transform(image=img, mask=mask)

        data["input"] = transformed["image"]
        data["target"] = transformed["mask"]
        return data


def get_train_augmentation(dataset, hyp: any, imgsz: int) -> Callable:
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
    )

    pre_transform = Compose([mosaic, affine])
    if hyp.copy_paste_mode == "flip":
        pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    else:
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
                p=hyp.copy_paste,
                mode=hyp.copy_paste_mode,
            )
        )

    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            CutMix(dataset, pre_transform=pre_transform, p=hyp.cutmix),
            Albumentations(p=1.0, transforms=getattr(hyp, "augmentations", None)),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr),
        ]
    )