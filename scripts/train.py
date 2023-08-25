from spotipy_torch import utils
from spotipy_torch.data import SpotsDataset
from spotipy_torch.model import Spotipy

from pathlib import Path
import lightning.pytorch as pl
from tormenter import transforms
from tormenter.pipeline import Pipeline

import configargparse
import torch

parser = configargparse.ArgumentParser(
    description="Train Spotipy model",
    config_file_parser_class=configargparse.YAMLConfigFileParser,
)
parser.add("-c", "--config", required=False, is_config_file=True, help="Config file path")
parser.add_argument("--data-dir", type=str, default="/data/spots/datasets/synthetic_clean")
parser.add_argument("--save-dir", type=str, default="/data/spots/results/synthetic_clean/spotipy_torch_v2")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--num-epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--pos-weight", type=float, default=10.0)
parser.add_argument("--backbone", type=str, default="unet")
parser.add_argument("--levels", type=int, default=4)
parser.add_argument("--crop-size", type=int, default=512)
parser.add_argument("--sigma", type=float, default=1.)
parser.add_argument("--mode", type=str, choices=["direct", "fpn"], default="direct")
parser.add_argument("--initial-fmaps", type=int, default=32)
parser.add_argument("--wandb-user", type=str, default="albertdm99")
parser.add_argument("--wandb-project", type=str, default="spotipy")
parser.add_argument("--loss", type=str, choices=["bce", "mse", "smoothl1", "adawing"], default="bce")
parser.add_argument("--skip-logging", action="store_true", default=False)
parser.add_argument("--kernel-size", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--convs-per-level", type=int, default=3)
parser.add_argument("--dropout", type=float, default=0)
parser.add_argument("--augment-prob", type=float, default=0.5)
parser.add_argument("--skip-bg-remover", action="store_true", default=False)
parser.add_argument("--dry-run", action="store_true", default=False)
args = parser.parse_args()


pl.seed_everything(args.seed, workers=True)
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
# torch.mps.manual_seed(args.seed)
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)

run_name = utils.get_run_name(args)

assert 0. <= args.augment_prob <= 1., f"Augment probability must be between 0 and 1, got {args.augment_prob}!"

augmenter = Pipeline()
augmenter.add(transforms.Crop(probability=1., size=(args.crop_size, args.crop_size)))
augmenter.add(transforms.FlipRot90(probability=args.augment_prob))
augmenter.add(transforms.Rotation(probability=args.augment_prob, order=1))
augmenter.add(transforms.IsotropicScale(probability=args.augment_prob, order=1, scaling_factor=(.5, 2.)))
augmenter.add(transforms.GaussianNoise(probability=args.augment_prob, sigma=(0, 0.05)))
augmenter.add(transforms.IntensityScaleShift(probability=args.augment_prob, scale=(0.5, 2.), shift=(-0.2, 0.2)))
        
# Load data
train_ds = SpotsDataset.from_folder(
    path=Path(args.data_dir)/"train",
    downsample_factors=[2**lv for lv in range(args.levels)],
    augmenter=augmenter,
    sigma=args.sigma,
    mode="max",
    normalizer=lambda img: utils.normalize(img, 1, 99.8),
)

val_ds = SpotsDataset.from_folder(
    path=Path(args.data_dir)/"val",
    downsample_factors=[2**lv for lv in range(args.levels)],
    augmenter=None,
    sigma=args.sigma,
    mode="max",
    normalizer=lambda img: utils.normalize(img, 1, 99.8),
)

train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

val_dl = torch.utils.data.DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

# Create model
model = Spotipy(
    args.backbone,
    backbone_params={
        "in_channels": 1,
        "initial_fmaps": args.initial_fmaps,
        "downsample_factors": tuple((2, 2) for _ in range(args.levels)),
        "kernel_sizes": tuple((args.kernel_size, args.kernel_size) for _ in range(args.convs_per_level)),
        "padding": "same",
    },
    levels=args.levels,
    mode=args.mode,
    background_remover=not args.skip_bg_remover,
    device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# model = torch.compile(model)

callbacks = []

if not args.dry_run:
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath=Path(args.save_dir)/f"{run_name}",
            filename="best",
            save_top_k=1,
            mode="min",
            save_last=True,
            save_weights_only=True,
        )
    )

logger = pl.loggers.WandbLogger(
    name=run_name,
    project=args.wandb_project,
    entity=args.wandb_user,
    save_dir=Path(args.save_dir)/f"{run_name}"/"wandb",
) if not args.skip_logging else None


accelerator = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
trainer = pl.Trainer(
    accelerator=accelerator,
    devices=1 if accelerator != "cpu" else "auto",
    logger=logger,
    callbacks=callbacks,
    deterministic=True,
    benchmark=False,
    max_epochs=args.num_epochs,
)



trainer.fit(
    model,
    train_dl,
    val_dl,
)

# Train model
# model.fit(
    # train_ds=train_ds,
    # val_ds=val_ds,
    # params=dict(vars(args), **{"run_name": run_name}),
# )
