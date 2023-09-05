from types import SimpleNamespace
from spotipy_torch import utils
from spotipy_torch.data import SpotsDataset
from spotipy_torch.model import (
    Spotipy,
    SpotipyModelCheckpoint,
    SpotipyModelConfig,
    SpotipyTrainingConfig,
)

from pathlib import Path
import lightning.pytorch as pl
from tormenter import transforms
from tormenter.pipeline import Pipeline

import configargparse
import torch


def get_run_name(args: SimpleNamespace):
    name = f"{Path(args.data_dir).stem}"
    name += f"_{args.backbone}"
    name += f"_lvs{int(args.levels)}"
    name += f"_fmaps{int(args.initial_fmaps)}"
    name += f"_{args.mode}"
    name += f"_convperlv{int(args.convs_per_level)}"
    name += f"_loss{args.loss}"
    name += f"_epochs{int(args.num_epochs)}"
    name += f"_lr{args.lr}"
    name += f"_bsize{int(args.batch_size)}"
    name += f"_ksize{int(args.kernel_size)}"
    name += f"_sigma{int(args.sigma)}"
    name += f"_crop{int(args.crop_size)}"
    name += f"_posweight{int(args.pos_weight)}"
    name += f"_seed{int(args.seed)}"
    name += f"_aug{args.augment_prob:.1f}"
    name += "_tormenter"  # !
    name += "_skipbgremover" if args.skip_bg_remover else ""
    name += "_dry" if args.dry else ""
    name = name.replace(".", "_")  # Remove dots to avoid confusion with file extensions
    return name


if __name__ == "__main__":

    parser = configargparse.ArgumentParser(
        description="Train Spotipy model",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add(
        "-c", "--config", required=False, is_config_file=True, help="Config file path"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/data/spots/datasets/synthetic_clean",
        help="Path to the data directory",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/data/spots/results/synthetic_clean/spotipy_torch_v2",
        help="Path to the save directory",
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Maximum number of files to use"
    )
    # model
    parser.add_argument(
        "--backbone",
        type=str,
        default="unet",
        choices=["unet", "resnet"],
        help="Backbone to use",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct", "fpn"],
        default="direct",
        help="Mode to use for the model",
    )
    parser.add_argument(
        "--in-channels", type=int, default=1, help="Number of input channels"
    )
    parser.add_argument(
        "--levels", type=int, default=4, help="Number of levels in the model"
    )
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument(
        "--initial-fmaps",
        type=int,
        default=32,
        help="Number of feature maps in the first layer",
    )
    parser.add_argument(
        "--fmap-inc-factor",
        type=float,
        default=2,
        help="Factor by which the number of feature maps increases per level",
    )
    parser.add_argument(
        "--kernel-size", type=int, default=3, help="Convolution kernel size"
    )
    parser.add_argument(
        "--convs-per-level",
        type=int,
        default=3,
        help="Number of convolutions per level",
    )
    parser.add_argument("--dropout", type=float, default=0, help="Dropout probability")
    parser.add_argument(
        "--skip-bg-remover",
        action="store_true",
        default=False,
        help="If given, won't use a background remover module",
    )
    # training
    parser.add_argument("--crop-size", type=int, default=512, help="Size of the crops")
    parser.add_argument(
        "--sigma", type=float, default=1.0, help="Sigma for the gaussian kernel"
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["bce", "mse", "smoothl1", "adawing"],
        default="bce",
        help="Loss function to use",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--augment-prob",
        type=float,
        default=0.5,
        help="Probability of applying each augmentation",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--batch-norm",
        action="store_true",
        default=False,
        help="If given, will use batch normalization",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=200, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of data workers"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--lr-reduce-patience",
        type=int,
        default=10,
        help="Learning rate scheduler patience",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=10.0,
        help="Weight for pixels containing a spot for the highest resolution loss function",
    )
    parser.add_argument(
        "--smart-crop",
        action="store_true",
        default=False,
        help="If given, random cropping will prioritize crops containing points",
    )
    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=None,
        help="Number of training samples per epoch (use all if None)",
    )
    # logging
    parser.add_argument(
        "--logger",
        type=str,
        choices=["none", "tensorboard", "wandb"],
        default="wandb",
        help="Logger to use. Unused if --skip-logging is set.",
    )
    parser.add_argument(
        "--wandb-user", type=str, default="albertdm99", help="Wandb user name"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="spotipy", help="Wandb project name"
    )
    parser.add_argument(
        "--skip-logging",
        action="store_true",
        default=False,
        help="If given, won't log to any logger",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        default=False,
        help="If given, won't save any output files",
    )

    args = parser.parse_args()

    if args.in_channels > 1:
        args.skip_bg_remover = True

    pl.seed_everything(args.seed, workers=True)

    run_name = get_run_name(args)

    assert (
        0.0 <= args.augment_prob <= 1.0
    ), f"Augment probability must be between 0 and 1, got {args.augment_prob}!"

    augmenter = Pipeline()
    augmenter.add(
        transforms.Crop(
            probability=1.0,
            size=(args.crop_size, args.crop_size),
            smart=args.smart_crop,
        )
    )
    augmenter.add(transforms.FlipRot90(probability=args.augment_prob))
    augmenter.add(transforms.Rotation(probability=args.augment_prob, order=1))
    # augmenter.add(transforms.IsotropicScale(probability=args.augment_prob, order=1, scaling_factor=(.5, 2.)))
    # augmenter.add(transforms.GaussianNoise(probability=args.augment_prob, sigma=(0, 0.05)))
    augmenter.add(
        transforms.IntensityScaleShift(
            probability=args.augment_prob, scale=(0.5, 2.0), shift=(-0.2, 0.2)
        )
    )

    augmenter_val = Pipeline()
    augmenter_val.add(
        transforms.Crop(
            probability=1.0,
            size=(args.crop_size, args.crop_size),
            smart=args.smart_crop,
        )
    )

    data_dir = Path(args.data_dir)
    assert data_dir.exists(), f"Data directory {data_dir} does not exist!"

    model_dir = Path(args.save_dir) / f"{run_name}"

    training_config = SpotipyTrainingConfig(
        sigma=args.sigma,
        crop_size=args.crop_size,
        loss_f=args.loss,
        pos_weight=args.pos_weight,
        lr=args.lr,
        lr_reduce_patience=args.lr_reduce_patience,
        num_train_samples=args.num_train_samples,
        smart_crop=args.smart_crop,
        optimizer="adamw",
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )

    save_dir = Path(args.save_dir) / f"{run_name}"
    if not args.dry:
        save_dir.mkdir(parents=True, exist_ok=True)

    model_config = SpotipyModelConfig(
        backbone=args.backbone,
        in_channels=args.in_channels,
        out_channels=1,
        initial_fmaps=args.initial_fmaps,
        fmap_inc_factor=args.fmap_inc_factor,
        n_convs_per_level=args.convs_per_level,
        downsample_factor=args.downsample_factor,
        kernel_size=args.kernel_size,
        padding="same",
        levels=args.levels,
        mode=args.mode,
        background_remover=not args.skip_bg_remover,
        batch_norm=args.batch_norm,
        dropout=args.dropout,
    )

    # Create model
    model = Spotipy(model_config)

    # model = torch.compile(model)

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    if not args.dry:
        callbacks.append(
            SpotipyModelCheckpoint(
                model_dir,
                training_config,
                monitor="val_loss",
            )
        )

    logger = None
    if not args.dry and not args.skip_logging:
        if args.logger == "wandb":
            logger = pl.loggers.WandbLogger(
                name=run_name,
                project=args.wandb_project,
                entity=args.wandb_user,
                save_dir=save_dir / "logs",
                config={
                    "model": vars(model_config),
                    "train": vars(training_config),
                    "data": str(args.data_dir),
                    "model_dir": str(model_dir),
                },
            )
            logger.watch(model, log_graph=False)
        elif args.logger == "tensorboard":
            logger = pl.loggers.TensorBoardLogger(
                save_dir=Path(args.save_dir) / f"{run_name}" / "logs",
                name=run_name,
            )
        else:
            print(f"Unknown logger {args.logger}! Will not log to any logger.")

    accelerator = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(model_config)
    print(training_config)

    train_ds = SpotsDataset.from_folder(
        data_dir / "train",
        downsample_factors=[
            args.downsample_factor**lv for lv in range(model_config.levels)
        ],  # ! model.downsample_factors
        augmenter=augmenter,
        sigma=training_config.sigma,
        mode="max",
        max_files=args.max_files,
        normalizer=lambda img: utils.normalize(img, 1, 99.8),
    )

    val_ds = SpotsDataset.from_folder(
        data_dir / "val",
        downsample_factors=[
            args.downsample_factor**lv for lv in range(model_config.levels)
        ],
        augmenter=augmenter_val,
        sigma=training_config.sigma,
        mode="max",
        max_files=args.max_files,
        normalizer=lambda img: utils.normalize(img, 1, 99.8),
    )

    if args.num_epochs > 0:
        # Train model
        model.fit(
            train_ds,
            val_ds,
            training_config,
            logger=logger,
            accelerator=accelerator,
            devices=1 if accelerator != "cpu" else "auto",
            callbacks=callbacks,
            num_workers=args.num_workers,
            deterministic=True,
            benchmark=False,
        )
