from landslide.data import LandslideDataset, dataloader, load_dataset
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # Read a batch

    data = load_dataset("L4S")
    dataset = LandslideDataset(data["valid"], do_normalize=False)
    print(f"Dataset length: {len(dataset)}")
    loader = dataloader(dataset, batch_size=3, shuffle=False)

    for i, (img, mask) in enumerate(loader):
        plt.figure(figsize=(12, 6))
        img = img.cpu().numpy().transpose((0, 2, 3, 1))  # (B, C, H, W) -> (B, H, W, C)
        mask = (
            mask.cpu().numpy().transpose((0, 2, 3, 1))
        )  # (B, C, H, W) -> (B, H, W, C)
        bs = img.shape[0]
        for j in range(bs):
            plt.subplot(3, 2, 2 * j + 1)
            plt.imshow(img[j])
            plt.axis("off")
            plt.subplot(3, 2, 2 * j + 2)
            plt.imshow(mask[j])
            plt.axis("off")
