"""Sample script to train a Spotipy model.
"""

import argparse
import numpy as np
from pathlib import Path
from skimage import io
from itertools import chain

import torch
from spotipy_torch.model import Spotipy
from spotipy_torch import utils
import lightning.pytorch as pl

IMAGE_EXTENSIONS = ("tif", "tiff", "png", "jpg", "jpeg")


def get_data(data_dir):
    """Load data from data_dir."""
    img_files = sorted(tuple(chain(*tuple(data_dir.glob(f"*.{ext}") for ext in IMAGE_EXTENSIONS))))
    spots_files = sorted(data_dir.glob("*.csv"))

    images = tuple(io.imread(str(f)) for f in img_files)
    spots = tuple(utils.read_coords_csv(str(f)).astype(np.float32) for f in spots_files)
    return images, spots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default="/data/spots/datasets/hybiss_spots_v4")
    parser.add_argument("--save-dir", type=Path, default="/data/tmp/spotipy_simple_train_debug")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    device_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print("Loading training data...")
    train_images, train_spots = get_data(args.data_dir / "train")
    print(f"Training data loaded (N={len(train_images)}).")

    print("Loading validation data...")
    val_images, val_spots = get_data(args.data_dir / "val")
    print(f"Validation data loaded (N={len(val_images)}).")

    print("Instantiating model...")
    model = Spotipy().to(torch.device(device_str))

    print("Launching training...")
    model.fit(
        train_images,
        train_spots,
        val_images,
        val_spots,
        save_dir=args.save_dir,
        accelerator=device_str,
    )
    print("Done!")
