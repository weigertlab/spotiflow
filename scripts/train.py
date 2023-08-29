from spotipy_torch import utils
from spotipy_torch.data import SpotsDataset
from spotipy_torch.model import Spotipy
from spotipy_torch.model.config import SpotipyModelConfig, SpotipyTrainingConfig

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


model_dir = Path(args.save_dir)/f"{run_name}"
model_dir.mkdir(parents=True, exist_ok=True)

training_config = SpotipyTrainingConfig(
    data_dir=args.data_dir,
    model_dir=Path(args.save_dir)/f"{run_name}",
    sigma=args.sigma,
    crop_size=args.crop_size,
    loss_f=args.loss,
    pos_weight=args.pos_weight,
    lr=args.lr,
    optimizer="adamw",
    batch_size=args.batch_size,
    num_epochs=args.num_epochs,
)


model_config = SpotipyModelConfig(
    backbone=args.backbone,
    in_channels=1,
    out_channels=1,
    initial_fmaps=args.initial_fmaps,
    n_convs_per_level=args.convs_per_level,
    downsample_factor=2,
    kernel_size=args.kernel_size,
    padding="same",
    levels=args.levels,
    mode=args.mode,
    background_remover=not args.skip_bg_remover,
    batch_norm=False,
    dropout=args.dropout,
)

# Create model
model = Spotipy(model_config)

# model = torch.compile(model)

callbacks = [
    pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
]

if not args.dry_run:
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath=training_config.model_dir,
            filename="best",
            save_top_k=1,
            mode="min",
            save_last=True,
            save_weights_only=True,
        )
    )

if not args.skip_logging:
    logger = pl.loggers.WandbLogger(
        name=run_name,
        project=args.wandb_project,
        entity=args.wandb_user,
        save_dir=Path(args.save_dir)/f"{run_name}"/"wandb",
        config=
        {"model": vars(model_config),
         "train": vars(training_config)},
    )
    logger.watch(model, log_graph=False)
else:
    logger = None


accelerator = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(model_config)
print(training_config)

# Train model
model.fit(
    training_config,
    augmenter=augmenter,
    logger=logger,
    accelerator=accelerator,
    devices=1 if accelerator != "cpu" else "auto",
    callbacks=callbacks,
    deterministic=True,
    benchmark=False,
)
