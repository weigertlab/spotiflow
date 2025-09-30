import argparse
import logging
import sys
from itertools import chain
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import psutil
import torch
from dask.diagnostics import ProgressBar as DaskProgressBar
from tqdm.auto import tqdm

from .. import __version__
from ..model import Spotiflow
from ..utils import dask_normalize_mi_ma, imread_wrapped, infer_n_tiles, str2bool
from ..utils.fitting import signal_to_background

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)


ALLOWED_EXTENSIONS = ("tif", "tiff", "png", "jpg", "jpeg", "zarr")


# Argument parser
def get_args():
    parser = argparse.ArgumentParser(
        "spotiflow-predict", description="Predict spots in image(s) using Spotiflow."
    )

    required = parser.add_argument_group(
        title="Required arguments",
        description="Arguments required to run the prediction model",
    )
    required.add_argument(
        "data_path",
        type=Path,
        help="Path to image file or directory of image files. If a directory, will process all images in the directory.",
    )
    required.add_argument(
        "-pm",
        "--pretrained-model",
        type=str,
        required=False,
        default="general",
        help="Pretrained model name. Defaults to 'general'.",
    )
    required.add_argument(
        "-md",
        "--model-dir",
        type=str,
        required=False,
        default=None,
        help="Model directory to load. If provided, will override --pretrained-model.",
    )
    required.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=False,
        default="spotiflow_results",
        help="Output directory to write the CSV(s). If not provided, will create a 'spotiflow_results' subfolder in the current folder.",
    )

    predict = parser.add_argument_group(
        title="Prediction arguments",
        description="Arguments to change the behaviour of spotiflow during prediction. To keep the default behaviour, do not provide these arguments.",
    )
    predict.add_argument(
        "-t",
        "--probability-threshold",
        type=float,
        required=False,
        default=None,
        help="Probability threshold for peak detection. If None, will load the optimal one. Defaults to None.",
    )
    predict.add_argument(
        "-n",
        "--n-tiles",
        type=int,
        required=False,
        default=None,
        nargs=2,
        help="Number of tiles to split the image into. When None will be automatically calculated based on `--max-tile-size`. This parameter can be used to calculate spots on larger images.",
    )
    predict.add_argument(
        "--max-tile-size",
        type=int,
        nargs="+",
        required=False,
        default=None,
        help="Maximal tile size per dim for splitting the image into tiles. Defaults to (2048,2048) for 2d and (512,512,32) for 3d. Decrease if running out of memory.",
    )
    predict.add_argument(
        "-min",
        "--min-distance",
        type=int,
        required=False,
        default=1,
        help="Minimum distance between spots for NMS. Defaults to 1.",
    )
    predict.add_argument(
        "-eb",
        "--exclude-border",
        type=int,
        required=False,
        default=1,
        help="Exclude spots within this distance from the border. Defaults to 1.",
    )
    predict.add_argument(
        "-s",
        "--scale",
        type=float,
        required=False,
        default=None,
        help=" Scale factor to apply to the image. Defaults to None.",
    )
    predict.add_argument(
        "-sp",
        "--subpix",
        type=str2bool,
        default=True,
        help="Whether to use the stereographic flow to compute subpixel localization. If None, will deduce from the model configuration. Defaults to True.",
    )
    parser.add_argument(
        "-spr",
        "--subpix-radius",
        type=int,
        default=0,
        help="Radius of the flow region to consider around the heatmap peak. Defaults to 0 (no aggregation).",
    )
    predict.add_argument(
        "-p",
        "--peak-mode",
        type=str,
        required=False,
        default="fast",
        choices=["fast", "skimage"],
        help="Peak detection mode (can be either 'skimage' or 'fast', which is a faster custom C++ implementation). Defaults to 'fast'.",
    )
    predict.add_argument(
        "--estimate-params",
        type=str2bool,
        default=False,
        help="Estimate fit parameters of detected spots by Gaussian fitting (eg FWHM, intensity). Defaults to False.",
    )
    predict.add_argument(
        "-norm",
        "--normalizer",
        type=str,
        required=False,
        default="auto",
        help="Normalizer to use. If None, will use the default normalizer. Defaults to 'auto' (percentile-based normalization with p_min=1, p_max=99.8).",
    )
    predict.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        help="Print verbose output. Defaults to False.",
    )
    predict.add_argument(
        "-d",
        "--device",
        type=str,
        required=False,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run model on. Defaults to 'auto'.",
    )
    predict.add_argument(
        "-c",
        "--channels",
        nargs="+",
        type=int,
        required=False,
        default=None,
        help="List of channels to use for prediction. "
        "0 will be interpreted as the first channel, 1 as the second, etc. "
        "This is only relevant for multi-channel images. If None, will use all channels. Defaults to None.",
    )
    predict.add_argument(
        "-zc",
        "--zarr-component",
        type=str,
        default=None,
        help="Zarr component to predict on, if necessary. Defaults to None."
    )
    predict.add_argument(
        "-zcl",
        "--zarr-component-lowres",
        type=str,
        default=None,
        help="Zarr component to use for computing the normalization percentiles, if necessary. Pass a lower resolution component to massively speed up the computation. Defaults to None.",
    )

    utils = parser.add_argument_group(
        title="Utility arguments",
        description="Diverse utility arguments, e.g. I/O related.",
    )
    utils.add_argument(
        "--exclude-hidden-files",
        action="store_true",
        required=False,
        default=False,
        help="Exclude hidden files in the input directory. Defaults to False.",
    )

    args = parser.parse_args()
    return args


def _check_valid_input_shape(shape, config):
    ndim = 3 if config.is_3d else 2
    return (
        len(shape) == ndim or len(shape) == ndim + 1 and shape[-1] == config.in_channels
    )


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

    if model.config.in_channels > 1 and args.estimate_params:
        raise ValueError(
            "Estimating parameters is only supported for single-channel input models. Please set --estimate-params to False."
        )

    if model.config.in_channels > 1:
        log.warning(
            "The loaded model is multi-channel. Spot intensities will not be output."
        )

    # Try to compile model
    try:
        model = torch.compile(model)
    except RuntimeError as _:
        log.info("Could not compile model. Will proceed without compilation.")

    # Set out_dir
    out_dir = args.out_dir

    # Check if data_path is a file or directory
    # If it's a file , check if it is a valid image file
    if args.data_path.is_file() or args.data_path.suffix == ".zarr":
        assert args.data_path.suffix[1:] in ALLOWED_EXTENSIONS, (
            f"File {args.data_path} is not a valid image file. Allowed extensions are: {ALLOWED_EXTENSIONS}"
        )
        image_files = [args.data_path]
        if out_dir is None:
            out_dir = args.data_path.parent / "spotiflow_results"

    # If directory, get all image files in the directory
    elif args.data_path.is_dir():
        image_files = sorted(
            tuple(
                chain(
                    *tuple(
                        args.data_path.glob(f"*.{ext}") for ext in ALLOWED_EXTENSIONS
                    )
                )
            )
        )
        if args.exclude_hidden_files:
            image_files = tuple(f for f in image_files if not f.name.startswith("."))
        if len(image_files) == 0:
            raise ValueError(
                f"No valid image files found in directory {args.data_path}. Allowed extensions are: {ALLOWED_EXTENSIONS}"
            )
        if out_dir is None:
            out_dir = args.data_path / "spotiflow_results"
    else:
        raise ValueError(f"Path {args.data_path} does not exist!")

    # Create out_dir if it doesn't exist
    out_dir.mkdir(exist_ok=True, parents=True)

    # Predict spots in images and write to CSV
    images = []

    for f in image_files:
        img = imread_wrapped(f, args.channels, args.zarr_component)
        if model.config.in_channels == 1:
            if isinstance(img, da.Array):
                img = da.squeeze(img)
            else:
                img = np.squeeze(img)
        if not _check_valid_input_shape(img.shape, model.config):
            raise ValueError(
                f"image {f} has invalid shape {img.shape} for model with is_3d={model.config.is_3d} and {model.config.in_channels} input channels. The image shape should be either (Y,X,[C]) for a 2D model or (Z,Y,X,[C]) for a 3D model, where the [C] dimension is optional for single-channel inputs."
            )
        images.append(img)

    for img, fname in tqdm(
        zip(images, image_files), desc="Predicting", total=len(images), unit="img"
    ):
        if args.n_tiles is None:
            n_tiles = infer_n_tiles(
                img.shape[:2] if not model.config.is_3d else img.shape[:3],
                args.max_tile_size,
            )
        else:
            n_tiles = tuple(args.n_tiles)

        if args.verbose:
            log.info(f"Predicting spots in {fname} with {n_tiles=}")

        _subpix_arg = False if not args.subpix else args.subpix_radius

        if isinstance(img, da.Array):
            _available_mem = psutil.virtual_memory().available

            if img.nbytes < _available_mem / 2:
                log.info(f"Bringing Zarr in-memory (~{img.nbytes/1e9:.2f} GB)...")
                with DaskProgressBar(minimum=1.):
                    img = img.compute()
            else:
                log.warning(
                    f"Zarr is too large to bring in-memory (~{img.nbytes/1e9:.2f} GB). Will process it as a Dask array."
                )
                if args.normalizer == "auto" and args.zarr_component_lowres is not None:
                    log.info(
                        "Trying to compute normalization percentiles on lower resolution image."
                    )
                    img_lowres = imread_wrapped(
                        fname, args.channels, args.zarr_component_lowres
                    )
                    if img_lowres.nbytes < _available_mem / 2:
                        with DaskProgressBar(minimum=1.):
                            img_lowres = img_lowres.compute()
                        p1, p998 = np.percentile(
                            img_lowres, (1, 99.8)
                        )
                        args.normalizer = lambda x: dask_normalize_mi_ma(x, p1, p998)
                        del img_lowres
                    else:
                        log.warning(
                            "Given low-resolution Zarr component is too large to bring in-memory. Will compute normalization percentiles on the full resolution image. This will be slower."
                        )
                else:
                    log.warning(
                        "Normalization percentiles will be computed on the full resolution image. This will be slower."
                    )

        _is_spotiflow_input_dask = isinstance(img, da.Array)

        spots, details = model.predict(
            img,
            prob_thresh=args.probability_threshold,
            n_tiles=n_tiles,
            min_distance=args.min_distance,
            exclude_border=args.exclude_border,
            scale=args.scale,
            subpix=_subpix_arg,
            peak_mode=args.peak_mode,
            normalizer=args.normalizer,
            verbose=args.verbose,
            device=args.device,
            fit_params=args.estimate_params,
        )
        csv_columns = ("y", "x")
        if spots.shape[1] == 3:
            csv_columns = ("z",) + csv_columns
        df = pd.DataFrame(np.round(spots, 4), columns=csv_columns)

        if not _is_spotiflow_input_dask: # Details are not computed for Dask arrays
            if model.config.in_channels == 1:
                df["intensity"] = np.round(details.intens, 2)
            df["probability"] = np.round(details.prob, 3)
            if args.estimate_params:
                df["fwhm"] = np.round(details.fit_params.fwhm, 3)
                df["intens_A"] = np.round(details.fit_params.intens_A, 3)
                df["intens_B"] = np.round(details.fit_params.intens_B, 3)
                df["snb"] = np.round(signal_to_background(details.fit_params), 3)

        df.to_csv(out_dir / f"{fname.stem.replace('.ome', '')}.csv", index=False)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
