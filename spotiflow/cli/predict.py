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

# Argument parser
def get_args():
    parser = argparse.ArgumentParser("spotiflow-predict",
                                     description="Predict spots in image(s) using Spotiflow.")

    required = parser.add_argument_group(title="Required arguments",
                                         description="Arguments required to run the prediction model")
    required.add_argument("data_path",
                          type=Path,
                          help=f"Path to image file or directory of image files. If a directory, will process all images in the directory.")
    required.add_argument("-pm","--pretrained-model",
                          type=str, required=False, default="general",
                          help="Pretrained model name. Defaults to 'general'.")
    required.add_argument("-md", "--model-dir",
                          type=str, required=False, default=None,
                          help="Model directory to load. If provided, will override --pretrained-model.")
    required.add_argument("-o", "--out-dir",
                          type=Path, required=False, default=None,
                          help="Output directory. If not provided, will create a 'spotiflow_results' subfolder in the input folder and write the CSV(s) there.")

    predict = parser.add_argument_group(title="Prediction arguments",
                                        description="Arguments to change the behaviour of spotiflow during prediction. To keep the default behaviour, do not provide these arguments.")
    predict.add_argument("-t", "--probability-threshold",
                            type=float, required=False, default=None,
                            help="Probability threshold for peak detection. If None, will load the optimal one. Defaults to None.")
    predict.add_argument("-n", "--n-tiles",
                            type=int, required=False, default=(1, 1), nargs=2,
                            help="Number of tiles to split the image into. Defaults to (1, 1). This parameter can be used to calculate spots on larger images.")
    predict.add_argument("-min", "--min-distance",
                            type=int, required=False, default=1,
                            help="Minimum distance between spots for NMS. Defaults to 1.")
    predict.add_argument("-eb", "--exclude-border",
                            action="store_true", required=False,
                            help="Exclude spots within this distance from the border. Defaults to 0.")
    predict.add_argument("-s", "--scale",
                            type=int, required=False, default=None,
                            help=" Scale factor to apply to the image. Defaults to None.")
    predict.add_argument("-sp", "--subpix",
                            action="store_true", required=False,
                            help="Whether to use the stereographic flow to compute subpixel localization. If None, will deduce from the model configuration. Defaults to None.")
    predict.add_argument("-p", "--peak-mode",
                            type=str, required=False, default="fast", choices=["fast", "skimage"],
                            help="Peak detection mode (can be either 'skimage' or 'fast', which is a faster custom C++ implementation). Defaults to 'fast'.")
    predict.add_argument("-norm", "--normalizer",
                            type=str, required=False, default="auto",
                            help="Normalizer to use. If None, will use the default normalizer. Defaults to 'auto' (percentile-based normalization with p_min=1, p_max=99.8).")
    predict.add_argument("-v", "--verbose",
                            action="store_true", required=False,
                            help="Print verbose output. Defaults to False.")
    predict.add_argument("-d", "--device",
                            type=str, required=False, default="auto", choices=["auto", "cpu", "cuda", "mps"],
                            help="Device to run model on. Defaults to 'auto'.")

    utils = parser.add_argument_group(title="Utility arguments",
                                      description="Diverse utility arguments, e.g. I/O related.")
    utils.add_argument("--exclude-hidden-files", action="store_true", required=False, default=False, help="Exclude hidden files in the input directory. Defaults to False.")

    args = parser.parse_args()
    return args

def _imread_wrapped(fname):
    try:
        return imread(fname)
    except Exception as e:
        log.error(f"Could not read image {fname}. Execution will halt.")
        raise e

def main():
    # Get arguments from command line
    args = get_args()
    log.info(f"Spotiflow - version {__version__}")

    # Choose prediction method from_folder or from_pretrained
    if args.model_dir is not None:
        model = Spotiflow.from_folder(args.model_dir)
        log.info("Given local model loaded.")
    else:
        model = Spotiflow.from_pretrained(args.pretrained_model)

    # Try to compile model
    try:
        model = torch.compile(model)
    except RuntimeError as _:
        log.info("Could not compile model. Will proceed without compilation.")

    # Set out_dir
    out_dir = args.out_dir

    # Check if data_path is a file or directory
    # If it's a file , check if it is a valid image file
    if args.data_path.is_file():
        assert args.data_path.suffix[1:] in ALLOWED_EXTENSIONS, f"File {args.data_path} is not a valid image file. Allowed extensions are: {ALLOWED_EXTENSIONS}"
        image_files = [args.data_path]
        if out_dir is None:
            out_dir = args.data_path.parent/"spotiflow_results"

    # If directory, get all image files in the directory
    elif args.data_path.is_dir():
        image_files = sorted(
            tuple(chain(*tuple(args.data_path.glob(f"*.{ext}") for ext in ALLOWED_EXTENSIONS)))
        )
        if args.exclude_hidden_files:
            image_files = tuple(f for f in image_files if not f.name.startswith("."))
        if len(image_files) == 0:
            raise ValueError(f"No valid image files found in directory {args.data_path}. Allowed extensions are: {ALLOWED_EXTENSIONS}")
        if out_dir is None:
            out_dir = args.data_path/"spotiflow_results"
    else:
        raise ValueError(f"Path {args.data_path} does not exist!")

    # Create out_dir if it doesn't exist
    out_dir.mkdir(exist_ok=True, parents=True)

    # Predict spots in images and write to CSV
    images = [_imread_wrapped(img) for img in image_files]
    for img, fname in tqdm(zip(images, image_files), desc="Predicting", total=len(images)):
        spots, _ = model.predict(img,
                                 prob_thresh=args.probability_threshold,
                                 n_tiles=tuple(args.n_tiles),
                                 min_distance=args.min_distance,
                                 exclude_border=args.exclude_border,
                                 scale=args.scale,
                                 subpix=args.subpix,
                                 peak_mode=args.peak_mode,
                                 normalizer=args.normalizer,
                                 verbose=args.verbose,
                                 device=args.device,)
        write_coords_csv(spots, out_dir/f"{fname.stem}.csv")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
