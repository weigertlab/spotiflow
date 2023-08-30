from spotipy_torch import utils
from spotipy_torch.data import SpotsDataset
from spotipy_torch.model import Spotipy, SpotipyModelCheckpoint, SpotipyModelConfig, SpotipyTrainingConfig

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
parser.add_argument("--data-dir", type=str, default="/data/spots/datasets/synthetic_clean", help="Path to the data directory")
parser.add_argument("--save-dir", type=str, default="/data/spots/results/synthetic_clean/spotipy_torch_v2", help="Path to the save directory")
parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
parser.add_argument("--num-epochs", type=int, default=200, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--pos-weight", type=float, default=10.0, help="Weight for pixels containing a spot for the highest resolution loss function")
parser.add_argument("--backbone", type=str, default="unet", choices=["unet", "resnet"], help="Backbone to use")
parser.add_argument("--levels", type=int, default=4, help="Number of levels in the model")
parser.add_argument("--crop-size", type=int, default=512, help="Size of the crops")
parser.add_argument("--sigma", type=float, default=1., help="Sigma for the gaussian kernel")
parser.add_argument("--mode", type=str, choices=["direct", "fpn"], default="direct", help="Mode to use for the model")
parser.add_argument("--initial-fmaps", type=int, default=32, help="Number of feature maps in the first layer")
parser.add_argument("--wandb-user", type=str, default="albertdm99", help="Wandb user name")
parser.add_argument("--wandb-project", type=str, default="spotipy", help="Wandb project name")
parser.add_argument("--loss", type=str, choices=["bce", "mse", "smoothl1", "adawing"], default="bce", help="Loss function to use")
parser.add_argument("--skip-logging", action="store_true", default=False, help="If given, won't log to any logger")
parser.add_argument("--kernel-size", type=int, default=3, help="Convolution kernel size")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--convs-per-level", type=int, default=3, help="Number of convolutions per level")
parser.add_argument("--dropout", type=float, default=0, help="Dropout probability")
parser.add_argument("--augment-prob", type=float, default=0.5, help="Probability of applying an augmentation")
parser.add_argument("--skip-bg-remover", action="store_true", default=False, help="If given, won't use a background remover module")
parser.add_argument("--dry-run", action="store_true", default=False, help="If given, won't save any output files")
parser.add_argument("--logger", type=str, choices=["none", "tensorboard", "wandb"], default="wandb", help="Logger to use. Unused if --skip-logging is set.")
parser.add_argument("--smart-crop", action="store_true", default=False, help="If given, random cropping will prioritize crops containing points")
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
augmenter.add(transforms.Crop(probability=1., size=(args.crop_size, args.crop_size), smart=args.smart_crop))
augmenter.add(transforms.FlipRot90(probability=args.augment_prob))
augmenter.add(transforms.Rotation(probability=args.augment_prob, order=1))
augmenter.add(transforms.IsotropicScale(probability=args.augment_prob, order=1, scaling_factor=(.5, 2.)))
augmenter.add(transforms.GaussianNoise(probability=args.augment_prob, sigma=(0, 0.05)))
augmenter.add(transforms.IntensityScaleShift(probability=args.augment_prob, scale=(0.5, 2.), shift=(-0.2, 0.2)))



data_dir = Path(args.data_dir)
assert data_dir.exists(), f"Data directory {data_dir} does not exist!"

model_dir = Path(args.save_dir)/f"{run_name}"

training_config = SpotipyTrainingConfig(
    sigma=args.sigma,
    crop_size=args.crop_size,
    smart_crop=args.smart_crop,
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
        SpotipyModelCheckpoint(
            model_dir,
            training_config,
            monitor="val_loss",
        )
    )

save_dir = Path(args.save_dir)/f"{run_name}"
save_dir.mkdir(parents=True, exist_ok=True)

logger = None
if not args.skip_logging:
    if args.logger == "wandb":
        logger = pl.loggers.WandbLogger(
            name=run_name,
            project=args.wandb_project,
            entity=args.wandb_user,
            save_dir=save_dir/"logs",
            config=
            {"model": vars(model_config),
            "train": vars(training_config),
            "data": str(args.data_dir),
            "model_dir": str(model_dir)},
        )
        logger.watch(model, log_graph=False)
    elif args.logger == "tensorboard":
        logger = pl.loggers.TensorBoardLogger(
            save_dir=save_dir/"logs",
            name=run_name,
        )
    else:
        print(f"Unknown logger {args.logger}! Will not log to any logger.")


accelerator = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(model_config)
print(training_config)

train_ds = SpotsDataset.from_folder(
    data_dir/"train",
    downsample_factors=[2**lv for lv in range(model_config.levels)], # ! model.downsample_factors
    augmenter=augmenter,
    sigma=training_config.sigma,
    mode="max",
    normalizer=lambda img: utils.normalize(img, 1, 99.8)
)

val_ds = SpotsDataset.from_folder(
    data_dir/"val",
    downsample_factors=[2**lv for lv in range(model_config.levels)],
    augmenter=None,
    sigma=training_config.sigma,
    mode="max",
    normalizer=lambda img: utils.normalize(img, 1, 99.8),
)

# Train model
model.fit(
    train_ds,
    val_ds,
    training_config,
    logger=logger,
    accelerator=accelerator,
    devices=1 if accelerator != "cpu" else "auto",
    callbacks=callbacks,
    deterministic=True,
    benchmark=False,
)
