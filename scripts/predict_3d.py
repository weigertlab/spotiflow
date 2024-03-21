"""Sample script to detect spots on a 3D model with Spotiflow.
"""

import argparse
import numpy as np
from pathlib import Path
from skimage import io

from spotiflow.model import Spotiflow, SpotiflowModelConfig
from spotiflow import utils
import lightning.pytorch as pl
import tifffile

IMAGE_EXTENSIONS = ("tif", "tiff", "png", "jpg", "jpeg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path, default=Path("/data/tmp/spotiflow_3d_debug/smfish3d_finetuned"))
    parser.add_argument("--channel", type=int, choices=[0,1], required=True)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--min-distance", type=int, default=1, required=True)
    args = parser.parse_args()

    assert args.input.exists(), f"Input file not found: {args.input}"
    assert args.model.exists(), f"Model not found: {args.model}"
    assert args.output.suffix == ".csv", f"Output file must have a .csv extension"


    print("Reading input data...")
    img = io.imread(str(args.input))
    if img.ndim == 4:
        img = img[args.channel]
    else:
        print("Ignoring channel argument as input is (Z,Y,X)")

    if args.debug:
        print("Debug mode. Will predict on crop")
        img = img[128:192, 1000:1256, 1000:1256]
    print("Loading model...")
    model = Spotiflow.from_folder(args.model, map_location="auto")

    print(f"Image shape is: {img.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Predicting volume...")
    spots, details = model.predict(
        img,
        subpix=True,
        # n_tiles=(3, 10, 8) if not args.debug else (1, 1, 1),
        n_tiles=(1, 1, 1),
        device="auto",
        min_distance=args.min_distance,
    )

    if not args.debug:
        utils.write_coords_csv(spots, args.output)

    if args.debug:
        tifffile.imwrite(args.output.parent/"img_debug.tif", img)
