import argparse
from pathlib import Path

from landslide.data import dataset_write_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for landslide detection model training.")
    parser.add_argument("--dataset", type=str, help="Path to the dataset YAML file.", required=True)
    parser.add_argument("--batch", type=int, default=32, help="Batch size for data loader.")
    args = parser.parse_args()
    dataset_write_config(Path(args.dataset), batch_size=args.batch)
