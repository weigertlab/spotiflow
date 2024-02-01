from types import SimpleNamespace
from spotiflow import utils
from spotiflow.data import SpotsDataset
from spotiflow.model import (
    CustomEarlyStopping,
    Spotiflow,
    SpotiflowModelCheckpoint,
    SpotiflowModelConfig,
    SpotiflowTrainingConfig,
)

from pathlib import Path
import lightning.pytorch as pl
from spotiflow.augmentations import transforms
from spotiflow.augmentations.pipeline import Pipeline

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
    name += f"_mode_{args.mode}"
    name += f"_flow_{not args.skip_flow}"
    name += f"_bn_{not args.skip_batch_norm}"
    name += f"_aug{args.augment_prob:.1f}"
    name += "_tormenter"  # !
    name += "_skipbgremover" if not args.bg_remover else ""
    name += "_scaleaug" if args.scale_augmentation else ""
    name += "_finetuned" if args.pretrained_model else ""
    name += "_dry" if args.dry else ""
    name += "_single_scale_loss" if args.single_scale_loss else ""
    name += "_subpix" if not args.skip_flow else ""
    name = name.replace(".", "_")  # Remove dots to avoid confusion with file extensions
    return name


if __name__ == "__main__":

    parser = configargparse.ArgumentParser(
        description="Train Spotiflow model",
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
        default="/data/spots/results/synthetic_clean/spotiflow_v2",
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
        choices=["direct", "fpn", "slim"],
        default="slim",
        help="Mode to use for the model",
    )
    parser.add_argument(
        "--in-channels", type=int, default=1, help="Number of input channels"
    )
    parser.add_argument("--skip-flow", action="store_true")
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
        "--bg-remover",
        action="store_true",
        default=False,
        help="If given, will use a background remover module",
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
        "--skip-batch-norm",
        action="store_true",
        default=False,
        help="If given, will not use batch normalization",
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
    parser.add_argument(
        "--scale-augmentation",
        action="store_true",
        default=False,
        help="If given, add scale augmentation to the augment pipeline",
    )

    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="Path to a pretrained model to use for initialization",
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Early stopping patience (set to 0 to disable early stopping)",
    )

    parser.add_argument(
        "--min-epochs",
        type=int,
        default=30,
        help="Minimum number of epochs to be trained on. Only used if early stopping is enabled.",
    )

    parser.add_argument(
        "--single-scale-loss",
        action="store_true",
        default=False,
        help="Whether to use a single-scale loss function",
    )

    parser.add_argument(
        "--skip-train",
        action="store_true",
        default=False,
        help="Whether to skip training and only predict and evaluate on the test data",
    )

    parser.add_argument(
        "--skip-test",
        action="store_true",
        default=False,
        help="Whether to skip predicting and evaluating on the test data",
    )


    args = parser.parse_args()

    if args.in_channels > 1:
        args.bg_remover = False

    if args.pretrained_model is not None:
        assert Path(args.pretrained_model).exists(), "Given pretrained model does not exist!"

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
            point_priority=0.8 if args.smart_crop else 0,
        )
    )
    augmenter.add(transforms.FlipRot90(probability=args.augment_prob))
    augmenter.add(transforms.Rotation(probability=args.augment_prob, order=1))
    if args.scale_augmentation:
        augmenter.add(transforms.IsotropicScale(probability=args.augment_prob, order=1, scaling_factor=(.5, 2.)))
    augmenter.add(transforms.GaussianNoise(probability=args.augment_prob, sigma=(0, 0.05)))
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
            point_priority=0.8 if args.smart_crop else 0,
        )
    )

    data_dir = Path(args.data_dir)
    assert data_dir.exists(), f"Data directory {data_dir} does not exist!"

    model_dir = Path(args.save_dir) / f"{run_name}"

    training_config = SpotiflowTrainingConfig(
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
        finetuned_from=args.pretrained_model,
        early_stopping_patience=args.early_stopping_patience,
        loss_levels=None if not args.single_scale_loss else 1,
    )

    save_dir = Path(args.save_dir) / f"{run_name}"
    if not args.dry:
        save_dir.mkdir(parents=True, exist_ok=True)

    accelerator = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    if args.pretrained_model is not None:
        model = Spotiflow.from_pretrained(
            args.pretrained_model,
            inference_mode=False,
            map_location=accelerator,
        )
        print(f"Will finetune pretrained model loaded from {args.pretrained_model}")
        model_config = model.config
    else:
        model_config = SpotiflowModelConfig(
            backbone=args.backbone,
            in_channels=args.in_channels,
            out_channels=1,
            initial_fmaps=args.initial_fmaps,
            fmap_inc_factor=args.fmap_inc_factor,
            n_convs_per_level=args.convs_per_level,
            downsample_factor=args.downsample_factor,
            kernel_size=args.kernel_size,
            padding="same",
            compute_flow=not args.skip_flow,
            levels=args.levels,
            mode=args.mode,
            background_remover=args.bg_remover,
            batch_norm=not args.skip_batch_norm,
            dropout=args.dropout,
            sigma=args.sigma,
        )
        # Create model
        model = Spotiflow(model_config)

    try:
        model = torch.compile(model)
    except RuntimeError:
        print("Could not compile model. Proceeding without torch compilation.")


    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    if not args.dry:
        callbacks.append(
            SpotiflowModelCheckpoint(
                model_dir,
                training_config,
                monitor="val_loss",
            )
        )
    
    if training_config.early_stopping_patience > 0:
        print(f"Using early stopping: will stop training after validation loss is not improving for {training_config.early_stopping_patience} epochs")
        callbacks.append(
            CustomEarlyStopping(
                monitor="val_loss",
                patience=training_config.early_stopping_patience,
                mode="min",
                min_epochs=max(args.min_epochs-args.early_stopping_patience, 0),
                verbose=True
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

    print(model_config)
    print(training_config)

    effective_levels = model_config.levels if not args.single_scale_loss else 1
    train_ds = SpotsDataset.from_folder(
        data_dir / "train",
        downsample_factors=[
            args.downsample_factor**lv for lv in range(effective_levels)
        ],  # ! model.downsample_factors
        augmenter=augmenter,
        sigma=model_config.sigma,
        mode="max",
        compute_flow=not args.skip_flow,
        max_files=args.max_files,
        normalizer=lambda img: utils.normalize(img, 1, 99.8),
        random_state=args.seed,
    )

    val_ds = SpotsDataset.from_folder(
        data_dir / "val",
        downsample_factors=[
            args.downsample_factor**lv for lv in range(model_config.levels)
        ],
        augmenter=augmenter_val,
        sigma=model_config.sigma,
        mode="max",
        compute_flow=not args.skip_flow,
        max_files=args.max_files,
        normalizer=lambda img: utils.normalize(img, 1, 99.8),
        random_state=args.seed,
    )

    if args.num_epochs > 0 and not args.skip_train:
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
        del model

    if not args.skip_test:
        # Load the best checkpoint of the model we trained
        device = torch.device(accelerator)

        model = Spotiflow.from_folder(
            model_dir,
            inference_mode=True,
            map_location=accelerator,
            which="best",
        ).to(device)
        try:
            model = torch.compile(model)
        except RuntimeError:
            print("Could not compile model. Proceeding without torch compilation.")


        # Load the test data
        test_ds = SpotsDataset.from_folder(
            path=data_dir / "test",
            augmenter=None,
            downsample_factors=[2**lv for lv in range(model._levels)],
            sigma=1.,
            mode="max",
            normalizer=lambda img: utils.normalize(img, 1, 99.8),
            random_state=args.seed,
        )

        fnames = [Path(f).stem for f in test_ds.image_files]

        preds_out_dir = save_dir / "test_predictions"
        preds_out_dir.mkdir(exist_ok=True, parents=True)

        from tqdm.auto import tqdm
        test_preds = []
        for i, fname in tqdm(enumerate(fnames), desc="Predicting and writing", total=len(fnames)):
            normalized_img = test_ds.images[i]
            pts, _ = model.predict(
                normalized_img,
                min_distance=1,
                verbose=False,
                subpix=not args.skip_flow,
                device=device,
            )
            utils.write_coords_csv(pts, preds_out_dir/f"{fname}.csv")
            test_preds += [pts]
        
        # Compute the scores @3 and save
        metrics = utils.points_matching_dataset(
            test_ds._centers,
            test_preds,
            cutoff_distance=3.,
            by_image=False,
        )

        print("Test_metrics (Agg.):")
        print(metrics)

        metrics_save_dir = save_dir/"metrics"
        metrics_save_dir.mkdir(exist_ok=True, parents=True)
        import json

        with open(metrics_save_dir/"metrics.json", "w") as f:
            json.dump(vars(metrics), f)

        metrics_single = utils.points_matching_dataset(
            test_ds._centers,
            test_preds,
            cutoff_distance=3.,
            by_image=True,
        )

        print("Test metrics (single):")
        print(metrics_single)
        with open(metrics_save_dir/"metrics_single.json", "w") as f:
            json.dump(vars(metrics_single), f)
