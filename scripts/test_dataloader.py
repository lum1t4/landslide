from landslide.data import LandslideDataset, dataloader
from pathlib import Path
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # Read a batch
    ds = LandslideDataset(Path('./dataset/processed/L4S/valid'))
    print(f"Dataset length: {len(ds)}")
    loader = dataloader(ds, batch_size=3, shuffle=False)

    for i, (img, mask) in enumerate(loader):
        plt.figure(figsize=(12, 6))
        img = img.cpu().numpy().transpose((0, 2, 3, 1)) # (B, C, H, W) -> (B, H, W, C)
        mask = mask.cpu().numpy().transpose((0, 2, 3, 1)) # (B, C, H, W) -> (B, H, W, C)
        for j in range(3):
            plt.subplot(3, 2, 2*j+1)
            plt.imshow(img[j])
            plt.axis('off')
            plt.subplot(3, 2, 2*j+2)
            plt.imshow(mask[j])
            plt.axis('off')

