import argparse
import logging
from pathlib import Path
from itertools import chain
from tqdm.auto import tqdm

from skimage.io import imread

import torch

from .. import __version__
from ..model import Spotiflow
from ..utils import write_coords_csv

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

ALLOWED_EXTENSIONS = ("tif", "tiff", "png", "jpg", "jpeg")

def main():
    parser = argparse.ArgumentParser("spotiflow-predict", 
                                     description="Predict spots in image(s) using Spotiflow.")
    parser.add_argument("data_path", type=Path, help=f"Path to image file or directory of image files. If a directory, will process all images in the directory.")
    parser.add_argument("--pretrained-model", type=str, required=False, default="general", help="Pretrained model name. Defaults to 'general'.")
    parser.add_argument("--model-dir", type=str, required=False, default=None, help="Model directory to load. If provided, will override --pretrained-model.")
    parser.add_argument("--out-dir", type=Path, required=False, default=None, help="Output directory. If not provided, will create a 'spotiflow_results' subfolder in the input folder and write the CSV(s) there.")
    args = parser.parse_args()

    log.info(f"Spotiflow - version {__version__}")

    if args.model_dir is not None:
        model = Spotiflow.from_folder(args.model_dir)
        log.info("Given local model loaded.")
    else:
        model = Spotiflow.from_pretrained(args.pretrained_model)
    try:
        model = torch.compile(model)
    except RuntimeError as _:
        log.info("Could not compile model. Will proceed without compilation.")
    
    out_dir = args.out_dir

    if args.data_path.is_file():
        assert args.data_path.suffix[1:] in ALLOWED_EXTENSIONS, f"File {args.data_path} is not a valid image file. Allowed extensions are: {ALLOWED_EXTENSIONS}"
        image_files = [args.data_path]
        if out_dir is None:
            out_dir = args.data_path.parent/"spotiflow_results"
    
    elif args.data_path.is_dir():
        image_files = sorted(
            tuple(chain(*tuple(args.data_path.glob(f"*.{ext}") for ext in ALLOWED_EXTENSIONS)))
        )
        if len(image_files) == 0:
            raise ValueError(f"No valid image files found in directory {args.data_path}. Allowed extensions are: {ALLOWED_EXTENSIONS}")
        if out_dir is None:
            out_dir = args.data_path/"spotiflow_results"
    else:
        raise ValueError(f"Path {args.data_path} does not exist!")

    out_dir.mkdir(exist_ok=True, parents=True)

    images = [imread(img) for img in image_files]
    for img, fname in tqdm(zip(images, image_files), desc="Predicting", total=len(images)):
        spots, _ = model.predict(img, verbose=False)
        write_coords_csv(spots, out_dir/f"{fname.stem}.csv")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
