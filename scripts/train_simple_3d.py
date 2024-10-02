"""Sample script to train a Spotiflow model.
"""

import argparse
import logging
import sys
from itertools import chain
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
from skimage import io
from spotiflow import utils
from spotiflow.model import Spotiflow, SpotiflowModelConfig

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

IMAGE_EXTENSIONS = ("tif", "tiff", "png", "jpg", "jpeg")



def get_data(data_dir, debug=False):
    """Load data from data_dir."""
    img_files = sorted(tuple(chain(*tuple(data_dir.glob(f"*.{ext}") for ext in IMAGE_EXTENSIONS))))
    spots_files = sorted(data_dir.glob("*.csv"))
    if debug:
        img_files = img_files[:32]
        spots_files = spots_files[:32]
    images = tuple(io.imread(str(f)) for f in img_files)
    spots = tuple(utils.read_coords_csv3d(str(f)).astype(np.float32) for f in spots_files)
    return images, spots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default="/data/spots/datasets_3d/synth3d")
    parser.add_argument("--save-dir", type=Path, default="/data/tmp/spotiflow_3d_debug/synth3d")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--pretrained-path", type=Path, default=None)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--crop-size-depth", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pos-weight", type=float, default=10.)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    print("Loading training data...")
    train_images, train_spots = get_data(args.data_dir / "train", debug=args.debug)
    print(f"Training data loaded (N={len(train_images)}).")

    print("Loading validation data...")
    val_images, val_spots = get_data(args.data_dir / "val", debug=args.debug)
    print(f"Validation data loaded (N={len(val_images)}).")

    if args.pretrained_path is not None:
        print("Loading pretrained model...")
        model = Spotiflow.from_folder(args.pretrained_path)
        print("Launching fine-tuning...")
    else:
        print("Instantiating new model...")
        model = Spotiflow(SpotiflowModelConfig(in_channels=1, sigma=args.sigma, is_3d=True, levels=args.levels, grid=(1,1,1)))
        print("Launching training...")

    model.fit(
        train_images,
        train_spots,
        val_images,
        val_spots,
        save_dir=args.save_dir if not args.debug else args.save_dir/"debug",
        augment_train=True,
        device="auto",
        deterministic=False,
        logger="tensorboard" if not args.debug else "none",
        train_config={
            "num_epochs": args.num_epochs if not args.debug else 5,
            "crop_size": args.crop_size,
            "crop_size_depth": args.crop_size_depth,
            "smart_crop": True,
            "batch_size": args.batch_size,
            "pos_weight": args.pos_weight,
        }
    )
    print("Done!")
