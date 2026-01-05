import logging
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Literal, Optional, Sequence, Tuple, Union

import datetime
import dask.array as da
import lightning.pytorch as pl
import numpy as np
import pydash
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from csbdeep.internals.predict import tile_iterator
from scipy.ndimage import zoom
from tqdm.auto import tqdm
from typing_extensions import Self

from ..augmentations import transforms, transforms3d
from ..augmentations.pipeline import Pipeline as AugmentationPipeline
from ..data import Spots3DDataset, SpotsDataset
from ..utils import (
    center_crop,
    center_pad,
    filter_shape,
    flow_to_vector,
    infer_n_tiles,
    normalize,
    normalize_dask,
    points_matching_dataset,
    prob_to_points,
    spline_interp_points_2d,
    spline_interp_points_3d,
    subpixel_offset,
    estimate_params
)
from ..utils import (
    tile_iterator as parallel_tile_iterator,
)
from .backbones import ResNetBackbone, UNetBackbone
from .bg_remover import BackgroundRemover
from .config import SpotiflowModelConfig, SpotiflowTrainingConfig
from .post import FeaturePyramidNetwork, MultiHeadProcessor
from .pretrained import get_pretrained_model_path
from .trainer import SpotiflowModelCheckpoint, SpotiflowTrainingWrapper

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)


class Spotiflow(nn.Module):
    """Supervised spot detector using a multi-stage neural network as a backbone for
    feature extraction followed by resolution-dependent post-processing modules
    to allow loss computation and optimization at different resolution levels.
    """

    def __init__(self, config: Optional[SpotiflowModelConfig] = None) -> None:
        """Initialize the model.

        Args:
            config (Optional[SpotiflowModelConfig]): model configuration object. If None, will use the default configuration.
        """
        super().__init__()
        if config is None:
            log.info("No model config given, using default.")
            config = SpotiflowModelConfig()
            log.info(f"Default model config: {config}")
        self.config = config
        self._bg_remover = (
            BackgroundRemover() if config.background_remover else nn.Identity()
        )
        self._backbone = self._backbone_switcher()
        if config.mode == "direct":
            self._post = MultiHeadProcessor(
                in_channels_list=self._backbone.out_channels_list,
                out_channels=self.config.out_channels,
                kernel_sizes=self.config.kernel_sizes,
                initial_fmaps=self.config.initial_fmaps,
                fmap_inc_factor=self.config.fmap_inc_factor,
                use_slim_mode=False,
                is_3d=self.config.is_3d,
            )
        elif config.mode == "slim":
            self._post = MultiHeadProcessor(
                in_channels_list=self._backbone.out_channels_list,
                out_channels=self.config.out_channels,
                kernel_sizes=self.config.kernel_sizes,
                initial_fmaps=self.config.initial_fmaps,
                use_slim_mode=True,
                is_3d=self.config.is_3d,
            )
        elif config.mode == "fpn":
            self._post = FeaturePyramidNetwork(
                in_channels_list=self._backbone.out_channels_list,
                out_channels=self.config.out_channels,
            )
        else:
            raise NotImplementedError(f"Mode {config.mode} not implemented.")

        ConvModule = nn.Conv2d if not self.config.is_3d else nn.Conv3d
        BatchNormModule = nn.BatchNorm2d if not self.config.is_3d else nn.BatchNorm3d

        if self.config.compute_flow:
            self._flow = nn.Sequential(
                ConvModule(
                    self._backbone.out_channels_list[0],
                    self._backbone.out_channels_list[0],
                    3,
                    padding=1,
                    bias=not self.config.batch_norm,
                ),
                BatchNormModule(self._backbone.out_channels_list[0])
                if self.config.batch_norm
                else nn.Identity(),
                nn.ReLU(inplace=True),
                ConvModule(
                    self._backbone.out_channels_list[0],
                    self._backbone.out_channels_list[0],
                    3,
                    padding=1,
                    bias=not self.config.batch_norm,
                ),
                BatchNormModule(self._backbone.out_channels_list[0])
                if self.config.batch_norm
                else nn.Identity(),
                nn.ReLU(inplace=True),
                ConvModule(
                    self._backbone.out_channels_list[0],
                    3 * self.config.out_channels
                    if not self.config.is_3d
                    else 4 * self.config.out_channels,
                    3,
                    padding=1,
                ),
            )

        if self.config.is_3d and any(s != 1 for s in self.config.grid):
            self._downsampler = ConvModule(
                self.config.in_channels,
                self.config.initial_fmaps,
                kernel_size=(2 * s + 1 for s in self.config.grid),
                stride=self.config.grid,
                padding=self.config.grid,
                bias=not self.config.batch_norm,
            )
        else:
            self._downsampler = None

        self._levels = self.config.levels
        self._sigmoid = nn.Sigmoid()
        self._prob_thresh = [0.5] * self.config.out_channels

    @classmethod
    def from_folder(
        cls,
        pretrained_path: str,
        inference_mode=True,
        which: str = "best",
        map_location: Literal["auto", "cpu", "cuda", "mps"] = "auto",
        verbose: bool = False,
    ) -> Self:
        """Load a pretrained model.

        Args:
            pretrained_path (str): path to the model folder
            inference_mode (bool, optional): whether to set the model in eval mode. Defaults to True.
            which (str, optional): which checkpoint to load. Defaults to "best".
            map_location (str, optional): device string to load the model to. Defaults to 'auto' (hardware-based).

        Returns:
            Self: loaded model
        """
        assert Path(pretrained_path).exists(), f"Given path {pretrained_path} does not exist."
        if verbose:
            log.info(f"Loading model from folder: {pretrained_path}")
        model_config = SpotiflowModelConfig.from_config_file(
            Path(pretrained_path) / "config.yaml"
        )
        assert (
            map_location is not None
        ), "map_location must be one of ('auto', 'cpu', 'cuda', 'mps')"
        device = cls._retrieve_device_str(None, map_location, model_config)
        model = cls(model_config)
        model.load(
            pretrained_path,
            which=which,
            inference_mode=inference_mode,
            map_location=device,
        )
        return model.to(torch.device(device))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name: str,
        inference_mode: bool = True,
        which: str = "best",
        map_location: Literal["auto", "cpu", "cuda", "mps"] = "auto",
        cache_dir: Optional[Union[Path, str]] = None,
        verbose: bool = True,
        **kwargs,
    ) -> Self:
        """Load a pretrained model with given name

        Args:
            pretrained_name (str): name of the pretrained model to be loaded
            inference_mode (bool, optional): whether to set the model in eval mode. Defaults to True.
            which (str, optional): which checkpoint to load. Defaults to "best".
            map_location (str, optional): device string to load the model to. Defaults to 'auto' (hardware-based).
            cache_dir (Optional[Union[Path, str]], optional): directory to cache the model. Defaults to None. If None, will use the default cache directory (given by the env var SPOTIFLOW_CACHE_DIR if set, otherwise ~/.spotiflow).

        Returns:
            Self: loaded model
        """
        if verbose:
            log.info(f"Loading pretrained model: {pretrained_name}")
        if cache_dir is not None and isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        pretrained_path = get_pretrained_model_path(
            pretrained_name, cache_dir=cache_dir
        )
        if pretrained_path is not None:
            return cls.from_folder(
                pretrained_path,
                inference_mode=inference_mode,
                which=which,
                map_location=map_location,
                **kwargs,
            )
        else:
            raise ValueError(f"Pretrained model {pretrained_name} not found.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Forward pass

        Args:
            x:  torch.Tensor, input image

        Returns:
            out: Dict[torch.Tensor]
            out["heatmaps"]:  heatmaps at different resolutions. Highest resolution first.
            out["flow"]:      (n+1)d flow estimate
        """
        x = self._bg_remover(x)
        if self._downsampler is not None:
            x = self._downsampler(x)
        x = self._backbone(x)
        heatmaps = tuple(self._post(x))
        if self.config.compute_flow:
            flow = F.normalize(self._flow(x[0]), dim=1)
            return dict(heatmaps=heatmaps, flow=flow)
        else:
            return dict(heatmaps=heatmaps)

    def fit_dataset(
        self,
        train_ds: torch.utils.data.Dataset,
        val_ds: torch.utils.data.Dataset,
        train_config: SpotiflowTrainingConfig,
        accelerator: Literal["auto", "cpu", "cuda", "mps"] = "auto",
        logger: Optional[pl.loggers.Logger] = None,
        devices: Optional[int] = 1,
        num_workers: Optional[int] = 0,
        callbacks: Optional[Sequence[pl.callbacks.Callback]] = [],
        deterministic: Optional[bool] = True,
        default_root_dir: Optional[str] = None,
        benchmark: Optional[bool] = False,
    ):
        """Train the model using torch Datasets as input.

        Args:
            train_ds (torch.utils.data.Dataset): training dataset.
            val_ds (torch.utils.data.Dataset): validation dataset.
            train_config (SpotiflowTrainingConfig): training configuration
            accelerator (str): accelerator to use. Can be "auto" (automatically infered from available hardware), "cpu", "cuda", "mps".
            logger (Optional[pl.loggers.Logger], optional): logger to use. Defaults to None.
            devices (Optional[int], optional): number of accelerating devices to use. Defaults to 1.
            num_workers (Optional[int], optional): number of workers to use for data loading. Defaults to 0.
            callbacks (Optional[Sequence[pl.callbacks.Callback]], optional): callbacks to use during training. Defaults to no callbacks.
            deterministic (Optional[bool], optional): whether to use deterministic training. Set to True for deterministic behaviour at a cost of performance. Defaults to True.
            benchmark (Optional[bool], optional): whether to use benchmarking. Set to False for deterministic behaviour at a cost of performance. Defaults to False.
        """

        if self.config.is_3d and deterministic:
            log.warning(
                "Deterministic training is currently not supported in 3D mode. Disabling."
            )
            deterministic = False
        if not (train_ds._sigma == val_ds._sigma == self.config.sigma):
            raise ValueError(
                "Different sigma values given for training/validation data and model!"
            )


        if default_root_dir is None and hasattr(logger, 'save_dir'):
            default_root_dir = logger.save_dir

        
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=logger,
            callbacks=callbacks,
            deterministic=deterministic,
            benchmark=benchmark,
            enable_checkpointing=False,
            default_root_dir=default_root_dir,
            max_epochs=train_config.num_epochs,
            log_every_n_steps=min(50, len(train_ds) // train_config.batch_size),
        )
        training_wrapper = SpotiflowTrainingWrapper(self, train_config)
        train_dl, val_dl = training_wrapper.generate_dataloaders(
            train_ds,
            val_ds,
            num_train_samples=train_config.num_train_samples,
            num_workers=num_workers,
        )
        trainer.fit(training_wrapper, train_dl, val_dl)
        log.info("Training finished.")
        return

    def fit(
        self,
        train_images: Sequence[np.ndarray],
        train_spots: Sequence[np.ndarray],
        val_images: Sequence[np.ndarray],
        val_spots: Sequence[np.ndarray],
        augment_train: Union[bool, AugmentationPipeline] = True,
        save_dir: Optional[str] = None,
        train_config: Optional[Union[dict, SpotiflowTrainingConfig]] = None,
        device: Literal["auto", "cpu", "cuda", "mps"] = "auto",
        logger: Literal["none", "tensorboard", "wandb"] = "tensorboard",
        logging_name: Optional[str] = None,
        number_of_devices: Optional[int] = 1,
        num_workers: Optional[int] = 0,
        callbacks: Optional[Sequence[pl.callbacks.Callback]] = None,
        deterministic: Optional[bool] = True,
        benchmark: Optional[bool] = False,
        default_root_dir: Optional[str] = None,
        **dataset_kwargs,
    ):
        """Train a Spotiflow model.

        Args:
            train_images (Sequence[np.ndarray]): training images
            train_spots (Sequence[np.ndarray]): training spots
            val_images (Sequence[np.ndarray]): validation images
            val_spots (Sequence[np.ndarray]): validation spots
            augment_train (Union[bool, AugmentationPipeline]): whether to augment the training data. Defaults to True.
            save_dir (Optional[str], optional): directory to save the model to. Must be given if no checkpoint logger is given as a callback. Defaults to None.
            train_config (Optional[SpotiflowTrainingConfig], optional): training config. If not given, will use the default config. Defaults to None.
            device (Literal["cpu", "cuda", "mps"], optional): computing device to use. Can be "cpu", "cuda", "mps". Defaults to "cpu".
            logger (Optional[pl.loggers.Logger], optional): logger to use. Defaults to "tensorboard".
            logging_name (Optional[str], optional): name of the expriment name for the logger if applicable. If None or 'none', a random one will be generated. Defaults to None.
            number_of_devices (Optional[int], optional): number of accelerating devices to use. Only applicable to "cuda" acceleration. Defaults to 1.
            num_workers (Optional[int], optional): number of workers to use for data loading. Defaults to 0 (main process only).
            callbacks (Optional[Sequence[pl.callbacks.Callback]], optional): callbacks to use during training. Defaults to no callbacks.
            deterministic (Optional[bool], optional): whether to use deterministic training. Set to True for deterministic behaviour at a cost of performance. Defaults to True.
            benchmark (Optional[bool], optional): whether to use benchmarking. Set to False for deterministic behaviour at a cost of performance. Defaults to False.
            dataset_kwargs: additional arguments to pass to the SpotsDataset class. Defaults to no additional arguments.
        """
        # Make sure data is OK
        self._validate_fit_inputs(train_images, train_spots, val_images, val_spots)

        # Generate default training config if none is given
        if train_config is None:
            log.info("No training config given. Using default.")
            train_config = SpotiflowTrainingConfig()
        elif isinstance(train_config, dict):
            train_config = SpotiflowTrainingConfig(**train_config)
        elif isinstance(train_config, SpotiflowTrainingConfig):
            train_config = deepcopy(train_config)
        else:
            raise ValueError(f"Invalid training config: {train_config}")

        log.info(f"Training config is: {train_config}")

        # Avoid non consistent compute_flow/downsample_factors arguments (use the model instance values instead)
        if "compute_flow" in dataset_kwargs.keys():
            log.warning(
                "'compute_flow' argument given to Spotiflow.fit(). This argument is ignored."
            )
        if "downsample_factors" in dataset_kwargs.keys():
            log.warning(
                "'downsample_factors' argument given to Spotiflow.fit(). This argument is ignored."
            )
        if "sigma" in dataset_kwargs.keys():
            log.warning(
                "'sigma' argument given to Spotiflow.fit(). This argument is ignored."
            )
        dataset_kwargs["compute_flow"] = self.config.compute_flow
        dataset_kwargs["downsample_factors"] = [
            self.config.downsample_factor**lv for lv in range(self.config.levels)
        ]
        dataset_kwargs["sigma"] = self.config.sigma

        # Generate dataset kwargs for training and validation
        train_dataset_kwargs = deepcopy(dataset_kwargs)
        val_dataset_kwargs = deepcopy(pydash.omit(dataset_kwargs, ["augmenter"]))

        if not self.config.is_3d:
            _min_img_size = min(min(img.shape[:2]) for img in train_images)
        else:
            _min_img_size = min(min(img.shape[1:3]) for img in train_images)
        _min_crop_size = int(2 ** np.floor(np.log2(_min_img_size)))

        # Backbone min size
        _grid_exp = int(max(np.floor(np.log2(np.array(self.config.grid)))))
        _min_netw_size = int(2 ** (self.config.levels+_grid_exp))

        crop_size = (max(min(train_config.crop_size, _min_crop_size), _min_netw_size),) * 2

        if self.config.is_3d:
            _min_depth_size = min(img.shape[-3] for img in train_images)
            _min_crop_size_depth = int(2 ** np.floor(np.log2(_min_depth_size)))
            crop_size = (
                max(min(train_config.crop_size_depth, _min_crop_size_depth), _min_netw_size),
            ) + crop_size

        if train_config.smart_crop is True:
            point_priority = 0.8
        elif train_config.smart_crop is False:
            point_priority = 0.0
        else: 
            point_priority = train_config.smart_crop
        
        
        tr_augmenter = self.build_image_cropper(
            crop_size, point_priority=point_priority
        )
        
        # Build augmenters
        if isinstance(augment_train, AugmentationPipeline):
            tr_augmenter = tr_augmenter + augment_train
        elif augment_train is True:
            tr_augmenter = tr_augmenter + self.build_default_image_augmenter(
                point_priority=point_priority)
        elif augment_train is False:
            pass
        else:
            raise ValueError(f"Invalid augment_train value: {augment_train}")
            
        
        val_augmenter = self.build_image_cropper(
            crop_size, point_priority=point_priority
        )
        
        ActualSpotsDataset = SpotsDataset if not self.config.is_3d else Spots3DDataset
        # Generate datasets
        self.train_ds = ActualSpotsDataset(
            train_images,
            train_spots,
            augmenter=tr_augmenter,
            grid=self.config.grid,
            add_class_label=not self.config.is_3d,
            **train_dataset_kwargs,
        )
        self.val_ds = ActualSpotsDataset(
            val_images,
            val_spots,
            augmenter=val_augmenter,
            grid=self.config.grid,
            add_class_label=not self.config.is_3d,
            **val_dataset_kwargs,
        )

        # Add model checkpoint callback if not given (to save the model)
        if callbacks is None:
            callbacks = []

        if not any(isinstance(c, SpotiflowModelCheckpoint) for c in callbacks):
            if not save_dir:
                raise ValueError(
                    "save_dir argument must be given if no SpotiflowModelCheckpoint callback is given"
                )
            callbacks = [
                SpotiflowModelCheckpoint(
                    logdir=save_dir, train_config=train_config, monitor="val_loss"
                ),
                *callbacks,
            ]

        if logger == "tensorboard":
            logger = pl.loggers.TensorBoardLogger(save_dir=save_dir, name=f"spotiflow-{datetime.datetime.now().strftime('%Y%m%d_%H%M') if logging_name is None or logging_name == 'none' else logging_name}")
        elif logger == "wandb":
            Path(save_dir/"wandb").mkdir(parents=True, exist_ok=True)
            logger = pl.loggers.WandbLogger(save_dir=save_dir, project="spotiflow", name=f"{datetime.datetime.now().strftime('%Y%m%d_%H%M') if logging_name is None or logging_name == 'none' else logging_name}")
        else: 
            print(f'Using non standard logger {logger}')

        self.fit_dataset(
            self.train_ds,
            self.val_ds,
            train_config,
            accelerator=device,
            logger=logger,
            devices=number_of_devices,
            num_workers=num_workers,
            callbacks=callbacks,
            deterministic=deterministic,
            benchmark=benchmark,
            default_root_dir=default_root_dir,
        )
        return

    def _validate_fit_inputs(
        self, train_images, train_spots, val_images, val_spots
    ) -> None:
        """Validate the inputs given to the fit method."""
        for imgs, pts, split in zip(
            (train_images, val_images),
            (train_spots, val_spots),
            ("train", "validation"),
        ):
            assert len(imgs) == len(
                pts
            ), f"Number of images and points must be equal for {split} set"
            if not self.config.is_3d:
                assert all(
                    img.ndim in (2, 3) for img in imgs
                ), f"Images must be 2D (Y,X) or 3D (Y,X,C) for {split} set"
            else:
                assert all(
                    img.ndim in (3, 4) for img in imgs
                ), f"Images must be 3D (Z,Y,X) or 4D (Z,Y,X,C) for {split} set"
            if self.config.in_channels > 1:
                assert all(
                    img.ndim == (3 if not self.config.is_3d else 4)
                    and img.shape[-1] == self.config.in_channels
                    for img in imgs
                ), f"Input images should be in channel-last format (..., C) for {split} set"
            if not self.config.is_3d:
                assert all(
                    pts.ndim == 2 and pts.shape[1] in (2, 3) for pts in pts
                ), f"Points must be 2D (Y,X) for {split} set"
            else:
                assert all(
                    pts.ndim == 2 and pts.shape[1] == 3 for pts in pts
                ), f"Points must be 3D (Z,Y,X) for {split} set"
        return

    def build_image_cropper(
        self,
        crop_size: Union[Tuple[int, int], Tuple[int, int, int]],
        point_priority: float = 0.0,
    ):
        """Build default cropper for a dataset.

        Args:
            crop_size (Tuple[int, int]): tuple of (height, width) (if 2d) or (depth, height, width) (if 3d) to randomly crop the images.
            point_priority (float, optional): priority to sample regions containing spots when cropping. If 0, no priority is given (so all crops will be random). If 1, all crops will containg at least one spot. Defaults to 0.
        """
        cropper = AugmentationPipeline()
        if not self.config.is_3d:
            assert (
                len(crop_size) == 2
            ), "Crop size must be a 2-length tuple if mode is 2D."
            cropper.add(
                transforms.Crop(
                    probability=1.0, size=crop_size, point_priority=point_priority
                )
            )
        else:
            assert (
                len(crop_size) == 3
            ), "Crop size must be a 3-length tuple if mode is 3D."
            cropper.add(
                transforms3d.Crop3D(
                    probability=1.0, size=crop_size, point_priority=point_priority
                )
            )

        return cropper

    def build_default_image_augmenter(
        self,
        point_priority: float = 0.0,
    ) -> AugmentationPipeline:
        """Build default augmenter for training data.

        Args:
            point_priority (float, optional): priority to sample regions containing spots when cropping. Defaults to 0.
        """
        augmenter = AugmentationPipeline()
        
        if not self.config.is_3d:
            augmenter.add(transforms.FlipRot90(probability=0.5))
            augmenter.add(transforms.Rotation(probability=0.5, order=1))
            augmenter.add(transforms.GaussianNoise(probability=0.5, sigma=(0, 0.05)))
            augmenter.add(
                transforms.IntensityScaleShift(
                    probability=0.5, scale=(0.5, 2.0), shift=(-0.2, 0.2)
                )
            )
        else:
            augmenter.add(transforms3d.FlipRot903D(probability=1))
            augmenter.add(transforms3d.RotationYX3D(probability=0.5, order=1))
            augmenter.add(
                transforms3d.GaussianNoise3D(probability=0.5, sigma=(0, 0.05))
            )
            augmenter.add(
                transforms3d.IntensityScaleShift3D(
                    probability=0.85, scale=(0.5, 2.0), shift=(-0.2, 0.2)
                )
            )
        return augmenter

    def save(
        self,
        path: str,
        which: Literal["best", "last"] = "best",
        update_thresholds: bool = False,
    ) -> None:
        """Save the model to disk.

        Args:
            path (str): folder to save the model to
            which (Literal["best", "last"]): which checkpoint to save. Should be either "best" or "last".
            update_thresholds (bool, optional): whether to update the thresholds file. Defaults to False.
        """
        assert which in ("best", "last"), "which must be either 'best' or 'last'"
        checkpoint_path = Path(path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
            },
            str(checkpoint_path / f"{which}.pt"),
        )

        self.config.save(checkpoint_path / "config.yaml")
        if update_thresholds:
            thr_config_path = checkpoint_path / "thresholds.yaml"

            # Load existing thresholds file
            if thr_config_path.exists():
                with open(thr_config_path, "r") as fb:
                    threshold_dct = yaml.safe_load(fb)
            else:
                threshold_dct = {}

            # Update thresholds object according to current model and which checkpoint is updated
            threshold_dct.update({f"prob_thresh_{which}": self._prob_thresh})

            # Save the updated thresholds file
            with open(thr_config_path, "w") as fb:
                yaml.safe_dump(threshold_dct, fb, indent=4)
        return

    @staticmethod
    def cleanup_state_dict_keys(model_dict: OrderedDict) -> OrderedDict:
        """DEPRECATED. Remove the "model." prefix generated by the trainer wrapping class from the keys of a state dict.

        Args:
            model_dict (OrderedDict): state dictionary to clean

        Returns:
            OrderedDict: cleaned state dictionary
        """
        clean_dict = OrderedDict()
        for k, v in model_dict.items():
            if "model." in k:
                k = k.replace("model.", "")
            clean_dict[k] = v
        return clean_dict

    def load(
        self,
        path: str,
        which: Literal["best", "last"] = "best",
        inference_mode: bool = True,
        map_location: str = "cuda",
    ) -> None:
        """Load a model from disk.

        Args:
            path (str): folder to load the model from
            which (Literal['best', 'last'], optional): which checkpoint to load. Defaults to "best".
            inference_mode (bool, optional): whether to set the model in eval mode. Defaults to True.
            map_location (str, optional): device string to load the model to. Defaults to 'cuda'.
        """
        assert which in ("best", "last"), "which must be either 'best' or 'last'"
        thresholds_path = Path(path) / "thresholds.yaml"
        if thresholds_path.is_file():
            with open(thresholds_path, "r") as fb:
                thresholds = yaml.safe_load(fb)
        else:
            thresholds = {}

        states_path = Path(path) / f"{which}.pt"

        checkpoint = torch.load(
            states_path, map_location=map_location, weights_only=True
        )
        model_state = self.cleanup_state_dict_keys(checkpoint["state_dict"])
        self.load_state_dict(model_state)

        if f"prob_thresh_{which}" in thresholds.keys():
            self._prob_thresh = thresholds[f"prob_thresh_{which}"]
            if isinstance(self._prob_thresh, float):
                self._prob_thresh = [self._prob_thresh]
        else:  # ! For retrocompatibility, remove in the future
            self._prob_thresh = thresholds.get("prob_thresh", [0.5])
        if inference_mode:
            self.eval()
            return
        return

    def predict(
        self,
        img: Union[np.ndarray, da.Array],
        prob_thresh: Optional[float] = None,
        n_tiles: Tuple[int] = None,
        max_tile_size: int = None,
        min_distance: int = 1,
        exclude_border: bool = False,
        scale: Optional[int] = None,
        subpix: Optional[Union[bool, int]] = None,
        peak_mode: Literal["skimage", "fast"] = "fast",
        normalizer: Optional[Union[Literal["auto"], Callable]] = "auto",
        verbose: bool = True,
        progress_bar_wrapper: Optional[Callable] = None,
        device: Optional[
            Union[torch.device, Literal["auto", "cpu", "cuda", "mps"]]
        ] = None,
        distributed_params: Optional[dict] = None,
        fit_params: bool = False,
        use_tuned_tile_overlap: bool = False,
    ) -> Tuple[np.ndarray, SimpleNamespace]:
        """Predict spots in an image.

        Args:
            img (Union[np.ndarray, da.Array]): input image
            prob_thresh (Optional[float], optional): Probability threshold for peak detection. If None, will load the optimal one. Defaults to None.
            n_tiles (Tuple[int, int], optional): Number of tiles to split the image into. Defaults to (1,1).
            min_distance (int, optional): Minimum distance between spots for NMS. Defaults to 1.
            exclude_border (bool, optional): Whether to exclude spots at the border. Defaults to False.
            scale (Optional[int], optional): Scale factor to apply to the image. Defaults to None.
            subpix (bool, optional): Whether to use the stereographic flow to compute subpixel localization. If None, will deduce from the model configuration. Defaults to None.
            peak_mode (str, optional): Peak detection mode (can be either "skimage" or "fast", which is a faster custom C++ implementation). Defaults to "fast".
            normalizer (Optional[Union[Literal["auto"], callable]], optional): Normalizer to use. If None, will use the default normalizer. Defaults to "auto" (percentile-based normalization with p_min=1, p_max=99.8).
            verbose (bool, optional): Whether to print logs and progress. Defaults to True.
            progress_bar_wrapper (Optional[callable], optional): Progress bar wrapper to use. Defaults to None.
            device (Optional[Union[torch.device, Literal["auto", "cpu", "cuda", "mps"]]], optional): computing device to use. If None, will infer from model location. If "auto", will infer from available hardware. Defaults to None.
            fit_params (bool, optional): Whether to fit the model parameters to the input image. Defaults to False.
            use_tuned_tile_overlap (bool, optional): Whether to use tuned tile overlaps for prediction. Defaults to False. This behaviour will change in a future release.
        Returns:
            Tuple[np.ndarray, SimpleNamespace]: Tuple of (points, details). Points are the coordinates of the spots. Details is a namespace containing the spot-wise probabilities (`prob`), the heatmap (`heatmap`), the stereographic flow (`flow`), the 2D local offset vector field (`subpix`) and the spot intensities (`intens`).
        """
        if self.config.out_channels > 1:
            raise NotImplementedError(
                "Predicting with multiple channels is not supported yet."
            )

        if self.config.in_channels > 1 and fit_params:
            raise ValueError(
                "fit_params is not yet supported for multi-channel inputs. Please disable by setting 'fit_params=False'."
            )

        skip_details = isinstance(
            img, da.Array
        )  # Avoid computing details for non-NumPy inputs, which are assumed to be large

        if subpix is False:
            subpix_radius = -1
        elif subpix is True:
            subpix_radius = 0
        elif subpix is None:
            subpix_radius = 0 if self.config.compute_flow else -1
        else:
            subpix_radius = int(subpix)
        if not self.config.is_3d:
            assert img.ndim in (2, 3), "Image must be 2D (Y,X) or 3D (Y,X,C)"
        else:
            assert img.ndim in (3, 4), "Image must be 3D (Z,Y,X) or 4D (Z,Y,X,C)"

        if device is None or isinstance(device, str):
            device = self._retrieve_device_str(device)
            device = torch.device(device)
            if device is not None:
                self.to(device)

        if verbose:
            log.info(f"Will use device: {str(device)}")
            log.info(
                f"Predicting with prob_thresh = {self._prob_thresh if prob_thresh is None else prob_thresh}, min_distance = {min_distance}"
            )
            log.info(f"Peak detection mode: {peak_mode}")
            log.info(f"Image shape {img.shape}")

        if not self.config.is_3d and img.ndim == 2:
            img = img[..., None]
        elif self.config.is_3d and img.ndim == 3:
            img = img[..., None]

        if n_tiles is None:
            n_tiles = infer_n_tiles(
                img.shape[:-1], max_tile_size, device=next(self.parameters()).device
            )
        else: 
            n_tiles = tuple(n_tiles)
            
        if verbose:
            log.info(f"Predicting with {n_tiles} tiles")

        actual_n_dims = 2 if not self.config.is_3d else 3

        img = img.astype(np.float32)
        if scale is None or scale == 1:
            x = img
        else:
            if subpix_radius >= 0:
                raise NotImplementedError(
                    "Subpixel prediction is not supported yet when scale != 1."
                )
            if self.config.is_3d:
                raise NotImplementedError("3D scaling is not supported yet.")
            if verbose:
                log.info(f"Scaling image by factor {scale}")
            assert not (
                scale != 1 and isinstance(img, da.Array)
            ), "Dask arrays are not supported for scaling != 1"
            if scale < 1:
                # Make sure that the scaling can be inverted exactly at the same shape
                inv_scale = int(1 / scale)
                assert all(
                    s % inv_scale == 0 for s in img.shape[:2]
                ), "Invalid scale factor"
            factor = (scale, scale, 1)
            x = zoom(img, factor, order=1)

        div_by = tuple(
            self.config.downsample_factors[0][0] ** self.config.levels
            for _ in range(img.ndim - 1)
        ) + (1,)
        if self.config.is_3d and any(g > 1 for g in self.config.grid):
            div_by = tuple(g * d for g, d in zip((*self.config.grid, 1), div_by))
        if isinstance(normalizer, str) and normalizer == "auto":
            normalizer = normalize_dask if isinstance(x, da.Array) else normalize
        if normalizer is not None and callable(normalizer):
            if verbose:
                log.info("Normalizing...")
            x = normalizer(x)
        
        pad_shape = tuple(int(d * np.ceil(s / d)) for s, d in zip(x.shape, div_by))
        if verbose:
            log.info(f"Padding to shape {pad_shape}")
        x, padding = center_pad(x, pad_shape, mode="reflect")

        corr_grid = np.asarray(self.config.grid) if self.config.is_3d else 1
        out_shape = tuple(np.asarray(img.shape[:actual_n_dims]) // corr_grid)

        if np.any(corr_grid > 1):
            min_distance = min_distance / np.min(corr_grid)
            if verbose:
                log.info(f"Correcting internal min_distance to {min_distance} due to grid: {self.config.grid}.")
        self.eval()
        # Predict without tiling
        if all(n <= 1 for n in n_tiles):
            with torch.inference_mode():
                if isinstance(x, da.Array):
                    x = x.compute()
                img_t = torch.from_numpy(x).to(device).unsqueeze(0)  # Add B dimension
                if not self.config.is_3d:
                    img_t = img_t.permute(0, 3, 1, 2)  # BHWC -> BCHW
                else:
                    img_t = img_t.permute(0, 4, 1, 2, 3)  # BDHWC -> BCDHW
                out = self(img_t)

                y = (
                    self._sigmoid(out["heatmaps"][0].squeeze(0)).detach().cpu().numpy()
                )  # C'HW
                if subpix_radius >= 0:
                    if not self.config.is_3d:
                        flow = (
                            out["flow"][0]
                            .permute(1, 2, 0)
                            .detach()
                            .cpu()
                            .numpy()
                        )  # HW(3*C')
                    else:
                        flow = (
                            out["flow"][0]
                            .permute(1, 2, 3, 0)
                            .detach()
                            .cpu()
                            .numpy()
                        )  # HW(4*C')

            if scale is not None and scale != 1:
                y = zoom(y, (1.0, 1.0 / scale, 1.0 / scale), order=1)
            if subpix_radius >= 0:
                flow_cpt = flow.copy()

                _subpix = np.empty(
                    (self.config.out_channels,) + out_shape + (actual_n_dims,),
                    np.float32,
                )  # C'HW(2|3)
                flow = np.empty(
                    (self.config.out_channels,) + out_shape + (actual_n_dims + 1,),
                    np.float32,
                )  # C'HW(3|4)
                flow_dim = actual_n_dims + 1
                for cl in range(self.config.out_channels):
                    flow_curr = flow_cpt[..., cl * (flow_dim) : (cl + 1) * (flow_dim)]

                    _subpix_curr = flow_to_vector(
                        flow_curr,
                        sigma=self.config.sigma,
                    )

                    _subpix_curr = center_crop(_subpix_curr, out_shape)
                    _subpix[cl] = _subpix_curr

                    flow_curr = center_crop(flow_curr, out_shape)
                    flow[cl] = flow_curr

            ys = np.empty((self.config.out_channels,) + out_shape, np.float32)  # C'HW

            points = []
            probs = []
            for cl in range(self.config.out_channels):
                ys[cl] = center_crop(y[cl], out_shape)
                curr_pts = prob_to_points(
                    ys[cl],
                    prob_thresh=self._prob_thresh[cl]
                    if prob_thresh is None
                    else prob_thresh
                    if isinstance(prob_thresh, float)
                    else prob_thresh[0],
                    exclude_border=exclude_border,
                    mode=peak_mode,
                    min_distance=min_distance,
                )
                curr_probs = ys[cl][tuple(curr_pts.astype(int).T)].tolist()
                if subpix_radius >= 0:
                    subpix_tile = flow_to_vector(flow[cl], sigma=self.config.sigma)
                    _offset = subpixel_offset(
                        curr_pts, subpix_tile, ys[cl], radius=subpix_radius
                    )
                    curr_pts = curr_pts + _offset

                points.append(curr_pts)
                probs.append(curr_probs)

            assert (
                self.config.out_channels == 1
            ), "Trying to predict using a multi-channel network, which is not supported yet."
            # ! FIXME: This is a temporary fix which will stop working when multi-channel output is implemented
            points = points[0]
            probs = probs[0]
            y = ys[0]
            if subpix_radius >= 0:
                _subpix = _subpix[0]
                flow = flow[0]

        else:  # Predict with tiling
            padded_shape = tuple(np.array(x.shape[:actual_n_dims]) // corr_grid)
            if not skip_details:
                y = np.empty(padded_shape, np.float32)
                if subpix_radius >= 0:
                    _subpix = np.empty(padded_shape + (actual_n_dims,), np.float32)
                    flow = np.empty(padded_shape + (actual_n_dims + 1,), np.float32)
            points = []
            probs = []
            actual_n_tiles = n_tiles
            if not self.config.is_3d and x.ndim == 3 and len(n_tiles) == 2:
                actual_n_tiles = actual_n_tiles + (1,)
            elif self.config.is_3d and x.ndim == 4 and len(n_tiles) == 3:
                actual_n_tiles = actual_n_tiles + (1,)
            if not use_tuned_tile_overlap:
                # TODO: prev hardcoded defaults, should be changed to config-variable ones in future releases
                if self.config.is_3d:
                    n_block_overlaps = (2, 2, 2, 0)
                else:
                    n_block_overlaps = (4, 4, 0)
            else:
                n_block_overlaps = tuple(max(4, d//2) for d in div_by[:actual_n_dims]) + (0,)
                
                
            _n_tiles_progress = sum(1 for _ in tile_iterator(x, n_tiles=actual_n_tiles, block_sizes=div_by, n_block_overlaps=n_block_overlaps))
            if distributed_params is not None:
                gpu_id = distributed_params.get("gpu_id", 0)
                iter_tiles = parallel_tile_iterator(
                    x,
                    n_tiles=actual_n_tiles,
                    block_sizes=div_by,
                    n_block_overlaps=n_block_overlaps,
                    rank_id=gpu_id,
                    num_replicas=distributed_params.get(
                        "num_replicas", torch.cuda.device_count()
                    ),
                    dataloader_kwargs=distributed_params,
                )
            else:
                iter_tiles = tile_iterator(
                    x,
                    n_tiles=actual_n_tiles,
                    block_sizes=div_by,
                    n_block_overlaps=n_block_overlaps,
                )
            
            if verbose and callable(progress_bar_wrapper):
                iter_tiles = progress_bar_wrapper(iter_tiles, total=_n_tiles_progress)
            elif verbose:
                if distributed_params is None:
                    iter_tiles = tqdm(
                        iter_tiles,
                        desc="Predicting tiles",
                        total=_n_tiles_progress,
                    )
                else:
                    iter_tiles = tqdm(
                        iter_tiles,
                        desc=f"Predicting tiles (GPU {gpu_id})",
                        total=_n_tiles_progress
                        // distributed_params.get(
                            "num_replicas", torch.cuda.device_count()
                        ),
                        position=gpu_id,
                    )
            for item in iter_tiles:
                tile, s_src, s_dst = item
                if isinstance(tile, da.Array):
                    tile = tile.compute()
                with torch.inference_mode():
                    if isinstance(tile, np.ndarray):
                        tile = torch.from_numpy(tile)
                    img_t = tile.to(device).unsqueeze(0)  # Add B and C dimensions
                    if not self.config.is_3d:
                        img_t = img_t.permute(0, 3, 1, 2)  # BHWC -> BCHW
                    else:
                        img_t = img_t.permute(0, 4, 1, 2, 3)  # BDHWC -> BCDHW
                    out = self(img_t)
                    y_tile = self._sigmoid(out["heatmaps"][0].squeeze(0).squeeze(0))
                    y_tile = y_tile.detach().cpu().numpy()
                    p = prob_to_points(
                        y_tile,
                        prob_thresh=self._prob_thresh
                        if prob_thresh is None
                        else prob_thresh,
                        exclude_border=exclude_border,
                        mode=peak_mode,
                        min_distance=min_distance,
                    )
                    if self.config.is_3d and any(g > 1 for g in self.config.grid):
                        s_src_corr = tuple(
                            slice(s.start // g, s.stop // g, s.step)
                            for g, s in zip((*self.config.grid, 1), s_src)
                        )
                        s_dst_corr = tuple(
                            slice(s.start // g, s.stop // g, s.step)
                            for g, s in zip((*self.config.grid, 1), s_dst)
                        )
                    else:
                        s_src_corr, s_dst_corr = s_src, s_dst
                    # remove global offset
                    p -= np.array([s.start for s in s_src_corr[:actual_n_dims]])[None]
                    write_shape = tuple(
                        s.stop - s.start for s in s_dst_corr[:actual_n_dims]
                    )
                    p = filter_shape(p, write_shape, idxr_array=p)

                    y_tile_sub = y_tile[s_src_corr[:actual_n_dims]]
                    probs += y_tile_sub[tuple(p.astype(int).T)].tolist()
                    p_flow = p + np.array([s.start for s in s_src_corr[:actual_n_dims]])[None]
                    # add global offset
                    p += np.array([s.start for s in s_dst_corr[:actual_n_dims]])[None]
                    if not skip_details:
                        y[s_dst_corr[:actual_n_dims]] = y_tile_sub

                    # Flow
                    if subpix_radius >= 0:
                        permute_dims = (
                            (1, 2, 0) if not self.config.is_3d else (1, 2, 3, 0)
                        )
                        flow_tile = (
                            out["flow"][0]
                            .permute(*permute_dims)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        # Cartesian coordinates
                        subpix_tile = flow_to_vector(flow_tile, sigma=self.config.sigma)
                        _offset = subpixel_offset(
                            p_flow, subpix_tile, y_tile, radius=subpix_radius
                        )

                        p = p + _offset
                        if not skip_details:
                            flow_tile_sub = flow_tile[s_src_corr[:actual_n_dims]]
                            flow[s_dst_corr[:actual_n_dims]] = flow_tile_sub
                            _subpix[s_dst_corr[:actual_n_dims]] = subpix_tile[
                                s_src_corr[:actual_n_dims]
                            ]
                    points.append(p)
                    del out, img_t, tile, y_tile, p
                    torch.cuda.empty_cache()

            if scale is not None and scale != 1:
                y = zoom(y, (1.0 / scale, 1.0 / scale), order=1)
            if not skip_details:
                y = center_crop(y, out_shape)
            if subpix_radius >= 0 and not skip_details:
                flow = center_crop(flow, out_shape)
                _subpix = center_crop(_subpix, out_shape)

            points = np.concatenate(points, axis=0)

            # Remove padding
            padding_to_correct = (padding[0][0], padding[1][0])
            if self.config.is_3d:
                padding_to_correct = (*padding_to_correct, padding[2][0])
            points = points - np.array(padding_to_correct)[None] / corr_grid
        probs = np.asarray(probs)
        # if scale is not None and scale != 1:
        #     points = np.round((points.astype(float) / scale)).astype(int)
        probs = filter_shape(probs, out_shape, idxr_array=points)
        pts = filter_shape(points, out_shape, idxr_array=points)

        if self.config.is_3d and any(s > 1 for s in self.config.grid):
            pts *= np.asarray(self.config.grid)

        if skip_details:
            y = None
            _subpix = None
            flow = None
        elif subpix_radius < 0:
            _subpix = None
            flow = None

        if not skip_details and fit_params:
            fit_params = estimate_params(img[...,0], pts)
        else:
            fit_params = None
            
        if verbose:
            log.info(f"Found {len(pts)} spots")


        if not skip_details:
            # Retrieve intensity of the spots
            if (
                subpix_radius < 0
            ):  # no need to interpolate if subpixel precision is not used
                intens = img[tuple(pts.astype(int).T)]
            else:
                try:
                    _interp_fun = spline_interp_points_2d if not self.config.is_3d else spline_interp_points_3d
                    intens = _interp_fun(img, pts)
                except Exception as _:
                    log.warning(
                        "Spline interpolation failed to retrieve spot intensities. Will use nearest neighbour interpolation instead."
                    )
                    intens = img[tuple(pts.round().astype(int).T)]
        else:
            intens = None
        details = SimpleNamespace(
            prob=probs, heatmap=y, subpix=_subpix, flow=flow, intens=intens,
            fit_params=fit_params
            )
        return pts, details

    def predict_dataset(
        self,
        ds: torch.utils.data.Dataset,
        min_distance: int = 1,
        exclude_border: bool = False,
        batch_size: int = 4,
        prob_thresh: Optional[float] = None,
        return_heatmaps: bool = False,
        device: Optional[
            Union[torch.device, Literal["auto", "cpu", "cuda", "mps"], None]
        ] = "auto",
    ) -> Sequence[Tuple[np.ndarray, SimpleNamespace]]:
        """Predict spots from a SpotsDataset object.

        Args:
            ds (torch.utils.data.Dataset): dataset to predict
            min_distance (int, optional): Minimum distance between spots for NMS. Defaults to 1.
            exclude_border (bool, optional): Whether to exclude spots at the border. Defaults to False.
            batch_size (int, optional): Batch size to use for prediction. Defaults to 4.
            prob_thresh (Optional[float], optional): Probability threshold for peak detection. If None, will load the optimal one. Defaults to None.
            return_heatmaps (bool, optional): Whether to return the heatmaps. Defaults to False.
            device (Optional[Union[torch.device, Literal["auto", "cpu", "cuda", "mps"]]], optional): computing device to use. If None, will infer from model location. If "auto", will infer from available hardware. Defaults to "auto".

        Returns:
            Sequence[Tuple[np.ndarray, SimpleNamespace]]: Sequence of (points, details) tuples. Points are the coordinates of the spots. Details is a namespace containing the spot-wise probabilities, the heatmap and the 2D flow field.
        """
        preds = []
        dataloader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

        if device is None or isinstance(device, str):
            device = self._retrieve_device_str(device)
            device = torch.device(device)
            if device is not None:
                self.to(device)

        log.info(f"Will use device: {device}")

        device = torch.device(device)
        if self.config.out_channels == 1:
            log.info(
                f"Predicting with prob_thresh = {self._prob_thresh[0] if prob_thresh is None else prob_thresh if isinstance(prob_thresh, float) else prob_thresh[0]}, min_distance = {min_distance}"
            )
        else:
            raise NotImplementedError("Multichannel prediction not implemented yet.")
        self.eval()
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Predicting"):
                imgs = batch["img"].to(device)
                out = self(imgs)
                high_lv_preds = (
                    self._sigmoid(out["heatmaps"][0].squeeze(1)).detach().cpu().numpy()
                )
                for batch_elem in range(high_lv_preds.shape[0]):
                    preds += [high_lv_preds[batch_elem]]
                del out, imgs, batch
        if return_heatmaps:
            return preds
        p = [
            prob_to_points(
                pred,
                prob_thresh=self._prob_thresh[0]
                if prob_thresh is None
                else prob_thresh
                if isinstance(prob_thresh, float)
                else prob_thresh[0],
                exclude_border=exclude_border,
                min_distance=min_distance,
            )
            for pred in preds
        ]
        return p

    def predict_multichannel(
        self, img: np.ndarray, channels: Union[int, Tuple[int]] = None, **predict_kwargs
    ) -> Tuple[np.ndarray, Sequence[SimpleNamespace]]:
        """
        Wrapper function to retrieve spots in a multi-channel array independently per channel.
        Currently assumes that the array is in channel-last format ((Z)YXC).

        Args:
            img (np.ndarray): array to predict on. Should be in channel-last format ((Z)YXC).
            channels (Union[int, Tuple[int]], optional): indices of channels to predict on. If None, will predict on all channels. Defaults to None.
            **predict_kwargs: argument allows by Spotiflow.predict()

        Returns:
            Tuple[np.ndarray, Sequence[SimpleNamespace]: Detected spots and sequence of details. First item is a numpy array of shape (N, 3) or (N, 4) containing the coordinates of the spots with the channel they were found on as the last column. Each item in the second element correspond to the details,
                                                         which is a namespace containing the spot-wise probabilities, the heatmap and the 2D flow field.
        """
        log.info(
            "Data is assumed to be in channel-last format ((Z)YXC)."
        )  # TODO: needed?
        n_channels = img.shape[-1]
        if isinstance(channels, int):
            channels = (channels,)
        elif channels is None:
            channels = tuple(range(n_channels))

        assert all(
            c < n_channels for c in channels
        ), "All given channel indices should be smaller than the number of channels."
        all_details = []
        actual_n_dims = 3 if not self.config.is_3d else 4
        spots = np.empty((0, actual_n_dims))
        for c in tqdm(channels, desc="Predicting channels", total=len(channels)):
            curr_spots, details = self.predict(
                img[..., c], verbose=False, **predict_kwargs
            )
            curr_n_spots = curr_spots.shape[0]
            curr_spots = np.hstack(
                (curr_spots, c * np.ones((curr_n_spots, 1)))
            )  # Add channel indices as last column
            spots = np.vstack((spots, curr_spots))
            all_details += [details]

        return spots, all_details

    def optimize_threshold(
        self,
        val_ds: torch.utils.data.Dataset,
        cutoff_distance: int = None,
        min_distance: int = None,
        exclude_border: bool = False,
        threshold_range: Tuple[float, float] = (0.3, 0.7),
        niter: int = 11,
        batch_size: int = 1,
        device: Optional[
            Union[torch.device, Literal["auto", "cpu", "cuda", "mps"]]
        ] = None,
        subpix: Optional[bool] = None,
    ) -> None:
        """Optimize the probability threshold on an annotated dataset.
           The metric used to optimize is the F1 score.

        Args:
            val_ds (torch.utils.data.Dataset): dataset to optimize on
            cutoff_distance (int, optional): distance tolerance considered for points matching. Defaults to 3.
            min_distance (int, optional): Minimum distance between spots for NMS. Defaults to 1.. Defaults to 2.
            exclude_border (bool, optional): Whether to exclude spots at the border. Defaults to False.
            threshold_range (Tuple[float, float], optional): Range of thresholds to consider. Defaults to (.3, .7).
            niter (int, optional): number of iterations for both coarse- and fine-grained search. Defaults to 11.
            batch_size (int, optional): batch size to use. Defaults to 2.
            device (Optional[Union[torch.device, Literal["auto", "cpu", "cuda", "mps"]]], optional): computing device to use. If None, will infer from model location. If "auto", will infer from available hardware. Defaults to None.
            subpix (Optional[bool], optional): whether to use the stereographic flow to compute subpixel localization. If None, will deduce from the model configuration. Defaults to None.
        """
        if cutoff_distance is None:
            cutoff_distance = 2*self.config.sigma+1
        if min_distance is None:
            min_distance = self.config.sigma 
            
        val_hm_preds = []
        val_flow_preds = []
        val_gt_pts = []
        val_dataloader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
        )
        if device is None or isinstance(device, str):
            device = self._retrieve_device_str(device)
            device = torch.device(device)
            if device is not None:
                self.to(device)

        log.info(f"Will use device: {str(device)}")

        self.eval()
        with torch.inference_mode():
            for val_batch in val_dataloader:
                imgs = val_batch["img"].to(device)

                out = self(imgs)
                high_lv_hm_preds = (
                    self._sigmoid(out["heatmaps"][0]).detach().cpu().numpy()
                )
                if subpix:
                    curr_flow_preds = []
                    for flow in out["flow"]:
                        if not self.config.is_3d:
                            curr_flow_preds += [
                                flow
                                .permute(1, 2, 0)
                                .detach()
                                .cpu()
                                .numpy()
                            ]
                        else:
                            curr_flow_preds += [
                                flow
                                .permute(1, 2, 3, 0)
                                .detach()
                                .cpu()
                                .numpy()
                            ]

                for p in val_batch["pts"]:
                    val_gt_pts.append(p.numpy())

                val_flow_preds = []
                for batch_elem in range(high_lv_hm_preds.shape[0]):
                    val_hm_preds += [high_lv_hm_preds[batch_elem]]
                    if subpix:
                        flow_dim = 3 if not self.config.is_3d else 4
                        assert (
                            flow_dim * self.config.out_channels
                            == curr_flow_preds[batch_elem].shape[-1]
                        ), "Unexpected flow dimensions."
                        curr_val_flow_preds = np.concatenate(
                            [
                                flow_to_vector(
                                    curr_flow_preds[batch_elem][
                                        ..., cl * flow_dim : (cl + 1) * flow_dim
                                    ],
                                    sigma=self.config.sigma,
                                )[None, ...]
                                for cl in range(self.config.out_channels)
                            ],
                            axis=0,
                        )
                        val_flow_preds += [curr_val_flow_preds]
                del out, imgs, val_batch

        def _metric_at_threshold(thr, class_label: int = 0):
            val_pred_pts = [
                prob_to_points(
                    p[class_label],
                    prob_thresh=thr,
                    exclude_border=exclude_border,
                    min_distance=min_distance,
                )
                for p in val_hm_preds
            ]
            if subpix:
                val_pred_pts = [
                    pts + subpixel_offset(pts, np.squeeze(_subpix), np.squeeze(hmap), radius=0)
                    for hmap, pts, _subpix in zip(val_hm_preds, val_pred_pts, val_flow_preds)
                ]

            if self.config.is_3d and any(s > 1 for s in self.config.grid):
                corr_grid = np.asarray(self.config.grid) if self.config.is_3d else 1
                val_pred_pts = [pts * corr_grid for pts in val_pred_pts]

            # TODO: class label for 3D dataset
            stats = points_matching_dataset(
                val_gt_pts,
                val_pred_pts,
                cutoff_distance=cutoff_distance,
                by_image=True,
                class_label_p1=class_label if not self.config.is_3d else None,
            )
            return stats.f1

        def _grid_search(tmin, tmax, class_label: int = 0):
            thr = np.linspace(tmin, tmax, niter)
            ys = tuple(
                _metric_at_threshold(t, class_label)
                for t in tqdm(thr, desc="optimizing threshold")
            )
            i = np.argmax(ys)
            i1, i2 = max(0, i - 1), min(i + 1, len(thr) - 1)
            return thr[i], (thr[i1], thr[i2]), ys[i]

        best_thrs, best_f1s = [], []
        for cl in tqdm(
            range(self.config.out_channels),
            desc="Optimizing thresholds (class-wise)",
            disable=self.config.out_channels == 1,
        ):
            _, t_bounds, _ = _grid_search(*threshold_range, cl)
            best_thr, _, best_f1 = _grid_search(*t_bounds, cl)
            best_thrs += [float(best_thr)]
            best_f1s += [float(best_f1)]
        log.info(f"Best thresholds: {tuple(np.round(th, 3) for th in best_thrs)}")
        log.info(f"Best F1-score: {tuple(np.round(f1, 3) for f1 in best_f1s)}")
        self._prob_thresh = best_thrs
        return

    def _retrieve_device_str(
        self, device_str: Union[None, Literal["auto", "cpu", "cuda", "mps"]], model_config: Optional[SpotiflowModelConfig] = None
    ) -> str:
        """Retrieve the device string to use for the model.

        Args:
            device_str (Union[None, Literal["auto", "cpu", "cuda", "mps"]]): device string to use.
                If None, will use the location of the model parameters. If "auto", will infer from available hardware. Defaults to None.

        Returns:
            str: device string to use
        """
        _config = model_config if model_config is not None else self.config
        if device_str is not None and device_str not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError(
                f"device must be one of 'auto', 'cpu', 'cuda', 'mps', got {device_str}"
            )
        if device_str == "mps" and _config.is_3d:
            log.warning(
                "3D models are not supported by MPS as of now. Falling back to CPU."
            )
            return "cpu"
        if device_str is None:
            return str(next(self.parameters()).device)
        elif device_str == "auto":
            return (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available() and not _config.is_3d
                else "cpu"
            )
        return device_str

    def _backbone_switcher(self) -> nn.Module:
        """Switcher function to build the backbone."""
        if self.config.backbone == "unet":
            backbone_params = pydash.pick(
                self.config,
                "in_channels",
                "initial_fmaps",
                "fmap_inc_factor",
                "downsample_factors",
                "kernel_sizes",
                "batch_norm",
                "padding",
            )
            if (
                self.config.is_3d and any(g > 1 for g in self.config.grid)
            ):  # Gridded prediction, change in_channels to initial_fmaps, which is the output channels of the downsampler
                backbone_params["in_channels"] = backbone_params["initial_fmaps"]

            return UNetBackbone(
                concat_mode="cat", use_3d_convs=self.config.is_3d, **backbone_params
            )
        elif self.config.backbone == "unet_res":
            backbone_params = pydash.pick(
                self.config,
                "in_channels",
                "initial_fmaps",
                "fmap_inc_factor",
                "downsample_factors",
                "kernel_sizes",
                "batch_norm",
                "padding",
            )
            return UNetBackbone(concat_mode="add", **backbone_params)
        elif self.config.backbone == "resnet":
            backbone_params = pydash.pick(
                self.config,
                "in_channels",
                "initial_fmaps",
                "fmap_inc_factor",
                "downsample_factors",
                "kernel_sizes",
                "batch_norm",
                "padding",
            )
            return ResNetBackbone(**backbone_params)
        else:
            raise NotImplementedError(
                f"Backbone {self.config.backbone} not implemented."
            )
