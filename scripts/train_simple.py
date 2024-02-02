"""Sample script to train a Spotiflow model.
"""

import argparse
import numpy as np
from pathlib import Path
from skimage import io
from itertools import chain

from spotiflow.model import Spotiflow, SpotiflowModelConfig
from spotiflow import utils
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
    parser.add_argument("--save-dir", type=Path, default="/data/tmp/spotiflow_simple_train_debug")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    print("Loading training data...")
    train_images, train_spots = get_data(args.data_dir / "train")
    print(f"Training data loaded (N={len(train_images)}).")

    print("Loading validation data...")
    val_images, val_spots = get_data(args.data_dir / "val")
    print(f"Validation data loaded (N={len(val_images)}).")

    print("Instantiating model...")
    model = Spotiflow(SpotiflowModelConfig(sigma=args.sigma))

    print("Launching training...")
    model.fit(
        train_images,
        train_spots,
        val_images,
        val_spots,
        save_dir=args.save_dir,
        device="auto",
    )
    print("Done!")
