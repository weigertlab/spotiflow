import argparse
import logging
import sys
from itertools import chain
from pathlib import Path
from typing import Tuple

import lightning.pytorch as pl
import numpy as np
from skimage.io import imread

from .. import __version__
from ..model import Spotiflow, SpotiflowModelConfig
from ..model.pretrained import list_registered
from ..utils import read_coords_csv, read_coords_csv3d, str2bool

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

ALLOWED_EXTENSIONS = ("tif", "tiff", "png", "jpg", "jpeg")
PRETRAINED_MODELS = tuple(["general"]+sorted([m for m in list_registered() if m != "general"]))


def get_data(
    data_dir: Path, is_3d: bool = False
) -> Tuple[Tuple[np.ndarray], Tuple[np.ndarray]]:
    """Load data from given data_dir."""
    img_files = sorted(
        tuple(chain(*tuple(data_dir.glob(f"*.{ext}") for ext in ALLOWED_EXTENSIONS)))
    )
    spots_files = sorted(data_dir.glob("*.csv"))

    _read_spots_fun = read_coords_csv3d if is_3d else read_coords_csv
    images = tuple(imread(str(f)) for f in img_files)
    spots = tuple(_read_spots_fun(str(f)).astype(np.float32) for f in spots_files)
    return images, spots


# Argument parser
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "spotiflow-train",
        description="Train a Spotiflow model on annotated images of spots.",
    )

    required = parser.add_argument_group(
        title="Required arguments",
        description="Arguments required to train a Spotiflow model.",
    )
    required.add_argument(
        "data_dir",
        type=Path,
        help="Path to directory containing images and annotations. Please refer to the documentation (https://weigertlab.github.io/spotiflow/train.html#data-format) to see the required format.",
    )
        
    required.add_argument(
        "-o",
        "--outdir",
        type=Path,
        required=False,
        default="spotiflow_model",
        help="Output directory where the model will be stored (defaults to 'spotiflow_model' in current directory).",
    )

    model_args = parser.add_argument_group(
        title="Model configuration arguments",
        description="Arguments to configure the model architecture.",
    )
    model_args.add_argument(
        "--sigma",
        type=float,
        required=False,
        default=1.0,
        help="Sigma for the Gaussian heatmap generation. Set higher for detecting larger objects (e.g. blobs) or sparser objects. Defaults to 1.0.",
    )
    model_args.add_argument(
        "--levels",
        type=int,
        required=False,
        default=4,
        help="U-Net depth. Higher values will yield larger models with larger receptive fields. Defaults to 4.",
    )
    model_args.add_argument(
        "--grid",
        type=int,
        required=False,
        default=2,
        help="Gridding factor for downsampled prediction (g in the paper). Only used for 3D models. Defaults to 2.",
    )
    model_args.add_argument(
        "--is-3d",
        type=str2bool,
        required=False,
        default=False,
        help="Whether the model to be trained is 3D. Defaults to False (2D).",
    )
    model_args.add_argument(
        "--finetune-from",
        type=str,
        required=False,
        default=None,
        help="Path to a pre-trained model to finetune. If provided, the model will be loaded and trained on the given data. Note that all other model arguments will be ignored.",
    )
    model_args.add_argument(
        "--initial-fmaps",
        type=int,
        required=False,
        default=32,
        help="Number of feature maps in the first layer of the model. Lower values will yield smaller models, at the potential cost of performance. Defaults to 32.",
    )
    model_args.add_argument(
        "--in-channels",
        type=int,
        required=False,
        default=1,
        help="Number of input channels of your data. Defaults to 1.",
    )
    model_args.add_argument(
        "--fmap-inc-factor",
        type=int,
        required=False,
        default=2,
        help="Factor to increase the number of feature maps per level. Lower values will yield smaller models, at the potential cost of performance. Defaults to 2.",
    )
    train_args = parser.add_argument_group(
        title="Training arguments",
        description="Arguments to configure the training process.",
    )
    
    train_args.add_argument(
        "--subfolder",
        type=Path,
        nargs=2,
        required=False,
        default=['train', 'val'],
        help="Subfolder names for training and validation data. Defaults to ['train', 'val'].",
    )

    train_args.add_argument(
        "--train_samples",
        type=int,
        required=False,
        default=None,
        help="Number of training samples per epoch (defaults to None, which means all samples).",
    )

    train_args.add_argument(
        "--crop-size",
        type=int,
        required=False,
        default=512,
        help="Size of the random crops for training (in yx). Defaults to 512.",
    )
    train_args.add_argument(
        "--crop-size-z",
        type=int,
        required=False,
        default=32,
        help="Size of the random crops for training (in z). This argument is ignored for 2D training. Defaults to 32.",
    )
    train_args.add_argument(
        "--num-epochs",
        type=int,
        default=200,
        help="Number of epochs to train the model for. Defaults to 200.",
    )
    train_args.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate. Defaults to 3e-4."
    )
    train_args.add_argument(
        "--device",
        type=str,
        required=False,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to train the model on. Defaults to 'auto', which will infer based on the hardware.",
    )
    train_args.add_argument(
        "--augment",
        type=str2bool,
        required=False,
        default=True,
        help="Apply data augmentation during training. Defaults to True.",
    )
    
    train_args.add_argument(
        "--pos-weight",
        type=float,
        required=False,
        default=10.0,
        help="Positive weight for the loss function (lambda in the paper). Higher values can help the learning process for sparse datasets (few spots per image). Defaults to 10.",
    )
    train_args.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=4,
        help="Batch size for training. Defaults to 4.",
    )
    train_args.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Seed for reproducibility. Defaults to 42.",
    )
    train_args.add_argument(
        "--logger",
        type=str,
        required=False,
        choices=["none", "tensorboard", "wandb"],
        default="tensorboard",
        help="Logger to use for monitoring training. Defaults to 'tensorboard'.",
    )
    train_args.add_argument(
        "--smart-crop",
        type=str2bool,
        required=False,
        default=False,
        help="Use smart cropping for training. Defaults to False.",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    log.info(f"Spotiflow - version {__version__}")

    pl.seed_everything(args.seed, workers=True)

    log.info("Loading training data...")
    train_images, train_spots = get_data(args.data_dir / args.subfolder[0], is_3d=args.is_3d)
    if len(train_images) != len(train_spots):
        raise ValueError(f"Number of images and spots in {args.data_dir/'train'} do not match.")
    if len(train_images) == 0:
        raise ValueError(f"No images were found in the {args.data_dir/'train'}.")
    log.info(f"Training data loaded (N={len(train_images)}).")

    log.info("Loading validation data...")
    val_images, val_spots = get_data(args.data_dir / args.subfolder[1], is_3d=args.is_3d)
    if len(val_images) != len(val_spots):
        raise ValueError(f"Number of images and spots in {args.data_dir/'val'} do not match.")
    if len(val_images) == 0:
        raise ValueError(f"No images were found in the {args.data_dir/'val'}.")
    log.info(f"Validation data loaded (N={len(val_images)}).")

    if args.finetune_from is None:
        log.info("Instantiating model...")
        model = Spotiflow(
            SpotiflowModelConfig(
                sigma=args.sigma,
                is_3d=args.is_3d,
                initial_fmaps=args.initial_fmaps,
                fmap_inc_factor=args.fmap_inc_factor,
                levels=args.levels,
                in_channels=args.in_channels,
                grid=3 * (args.grid,) if args.is_3d else (1, 1),
            )
        )
    else:
        if args.finetune_from in PRETRAINED_MODELS:
            log.info(
                f"Loading pre-trained model '{args.finetune_from}' to be fine-tuned."
            )
            model = Spotiflow.from_pretrained(
                args.finetune_from,
                map_location=args.device,
                inference_mode=False,
                verbose=True,
            )
        else:
            if Path(args.finetune_from) == args.outdir:
                err_msg = "The save directory cannot be the same as the pre-trained model to be finetuned!"
                raise ValueError(err_msg)
            if not Path(args.finetune_from).is_dir():
                err_msg = f"Given pre-trained model '{args.finetune_from}' does not exist! Please provide either one of the pre-trained models ({', '.join(PRETRAINED_MODELS)}) or a valid directory containing a model.".strip().replace(
                    "\n", " "
                )
                raise ValueError(err_msg)
            log.info(f"Finetuning local model '{args.finetune_from}' to be fine-tuned.")
            model = Spotiflow.from_folder(
                args.finetune_from,
                map_location=args.device,
                inference_mode=False,
                verbose=True,
            )

    log.info("Launching training...")
    model.fit(
        train_images,
        train_spots,
        val_images,
        val_spots,
        save_dir=args.outdir,
        device=args.device,
        logger=args.logger,
        augment_train=args.augment,
        train_config={
            "batch_size": args.batch_size,
            "crop_size": args.crop_size,
            "crop_size_z": args.crop_size_z,
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "pos_weight": args.pos_weight,
            "num_train_samples":args.train_samples,
            "finetuned_from": args.finetune_from,
            "smart_crop": args.smart_crop,
        },
    )
    log.info("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
