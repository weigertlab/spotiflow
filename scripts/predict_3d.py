"""Sample script to detect spots on a 3D model with Spotiflow.
"""

import argparse
import logging
import sys
from pathlib import Path

import tifffile
from skimage import io
from spotiflow import utils
from spotiflow.model import Spotiflow
from spotiflow.model.pretrained import list_registered

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

IMAGE_EXTENSIONS = ("tif", "tiff", "png", "jpg", "jpeg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=str, default="smfish_3d")
    parser.add_argument("--channel", type=int, required=False, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--min-distance", type=int, default=1, required=True)
    parser.add_argument("--max-yx-tile-size", type=int, default=256)
    parser.add_argument("--max-z-tile-size", type=int, default=32)
    args = parser.parse_args()

    if args.model not in list_registered():
        args.model = Path(args.model)
        assert args.model.exists(), f"Model not found: {args.model}"
        
    assert args.input.exists(), f"Input file not found: {args.input}"
    assert args.output.suffix == ".csv", f"Output file must have a .csv extension"


    print("Reading input data...")
    img = io.imread(str(args.input))
    if img.ndim == 4:
        assert args.channel is not None, "Channel argument required if input is 4D (C,Z,Y,X)"
        img = img[args.channel]
    elif args.channel is not None:
        print("Ignoring channel argument as input is (Z,Y,X)")

    if args.debug:
        print("Debug mode. Will predict on crop")
        img = img[128:192, 1000:1256, 1000:1256]
    print("Loading model...")
    if args.model not in list_registered():
        model = Spotiflow.from_folder(args.model, map_location="auto")
    else:
        model = Spotiflow.from_pretrained(args.model, map_location="auto")
        

    print(f"Image shape is: {img.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_tiles = tuple(max(s//g, 1) for s, g in zip(img.shape, (args.max_z_tile_size, args.max_yx_tile_size, args.max_yx_tile_size)))
    print(n_tiles)
    print("Predicting volume...")
    spots, details = model.predict(
        img,
        subpix=True,
        n_tiles=n_tiles, # change if you run out of memory
        device="auto",
        min_distance=args.min_distance,
    )

    if not args.debug:
        utils.write_coords_csv(spots, args.output)

    if args.debug:
        tifffile.imwrite(args.output.parent/"img_debug.tif", img)
