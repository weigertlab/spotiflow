from spotipy_torch import datasets
from spotipy_torch.model import Spotipy

from pathlib import Path
import argparse
import torch





parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="/data/spots/datasets/synthetic_clean")
parser.add_argument("--save-dir", type=str, default="/data/spots/results/synthetic_clean/spotipy_torch_v2")
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--num-epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--pos-weight", type=float, default=10.0)
parser.add_argument("--backbone", type=str, default="resnet")
parser.add_argument("--levels", type=int, default=3)
parser.add_argument("--crop-size", type=int, default=512)
parser.add_argument("--sigma", type=float, default=1.)
parser.add_argument("--mode", type=str, choices=["direct", "fpn"], default="fpn")
parser.add_argument("--initial-fmaps", type=int, default=32)
parser.add_argument("--wandb-user", type=str, default="albertdm99")
parser.add_argument("--wandb-project", type=str, default="spotipy")
parser.add_argument("--skip-logging", action="store_true", default=False)
parser.add_argument("--run-name", type=str, required=False, default="synthetic_clean_test")
args = parser.parse_args()


# Load data
train_ds = datasets.AnnotatedSpotsDataset(Path(args.data_dir)/"train",
                                       downsample_factors=[2**lv for lv in range(args.levels)],
                                       sigma=args.sigma, 
                                       mode="max",
                                       augment_probability=.5,
                                       use_gpu=False,
                                       size=2*[args.crop_size],
                                       should_center_crop=False,
                                       norm_percentiles=(1, 99.8))

val_ds = datasets.AnnotatedSpotsDataset(Path(args.data_dir)/"val",
                                       downsample_factors=[2**lv for lv in range(args.levels)],
                                       sigma=args.sigma, 
                                       mode="max",
                                       augment_probability=0,
                                       use_gpu=False,
                                       size=None,
                                       norm_percentiles=(1, 99.8))

# Create model
model = Spotipy(
    args.backbone,
    backbone_params={
        "in_channels": 1,
        "initial_fmaps": args.initial_fmaps,
        "downsample_factors": tuple([1]+[2 for _ in range(args.levels-1)]),
        "kernel_sizes": tuple(3 for _ in range(args.levels)),
    },
    levels=args.levels,
    mode=args.mode,
    background_remover=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Train model
model.fit(
    train_ds=train_ds,
    val_ds=val_ds,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    params=vars(args),
)
