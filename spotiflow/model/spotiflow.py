import logging
from collections import OrderedDict
from copy import deepcopy
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Literal, Optional, Sequence, Tuple, Union
from typing_extensions import Self

import lightning.pytorch as pl
import numpy as np
import pydash
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from csbdeep.internals.predict import tile_iterator
from scipy.ndimage import zoom
from spotiflow.augmentations import transforms
from spotiflow.augmentations.pipeline import Pipeline as AugmentationPipeline
from tqdm.auto import tqdm

from ..data import SpotsDataset
from ..utils import (
    bilinear_interp_points,
    center_crop,
    center_pad,
    filter_shape,
    flow_to_vector,
    normalize,
    points_matching_dataset,
    prob_to_points,
)
from .backbones import ResNetBackbone, UNetBackbone
from .bg_remover import BackgroundRemover
from .config import SpotiflowModelConfig, SpotiflowTrainingConfig
from .post import FeaturePyramidNetwork, MultiHeadProcessor
from .pretrained import get_pretrained_model_path
from .trainer import SpotiflowModelCheckpoint, SpotiflowTrainingWrapper
from ..utils import subpixel_offset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


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
            )
        elif config.mode == "slim":
            self._post = MultiHeadProcessor(
                in_channels_list=self._backbone.out_channels_list,
                out_channels=self.config.out_channels,
                kernel_sizes=self.config.kernel_sizes,
                initial_fmaps=self.config.initial_fmaps,
                use_slim_mode=True,
            )
        elif config.mode == "fpn":
            self._post = FeaturePyramidNetwork(
                in_channels_list=self._backbone.out_channels_list,
                out_channels=self.config.out_channels,
            )
        else:
            raise NotImplementedError(f"Mode {config.mode} not implemented.")

        if self.config.compute_flow:
            self._flow = nn.Sequential(
                nn.Conv2d(
                    self._backbone.out_channels_list[0],
                    self._backbone.out_channels_list[0],
                    3,
                    padding=1,
                    bias=False if self.config.batch_norm else True,
                ),
                nn.BatchNorm2d(self._backbone.out_channels_list[0])
                if self.config.batch_norm
                else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self._backbone.out_channels_list[0],
                    self._backbone.out_channels_list[0],
                    3,
                    padding=1,
                    bias=False if self.config.batch_norm else True,
                ),
                nn.BatchNorm2d(self._backbone.out_channels_list[0])
                if self.config.batch_norm
                else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(self._backbone.out_channels_list[0], 3, 3, padding=1),
            )

        self._levels = self.config.levels
        self._sigmoid = nn.Sigmoid()
        self._prob_thresh = 0.5

    @classmethod
    def from_folder(
        cls,
        pretrained_path: str,
        inference_mode=True,
        which: str = "best",
        map_location: Literal["auto", "cpu", "cuda", "mps"] = "auto",
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
        model_config = SpotiflowModelConfig.from_config_file(
            Path(pretrained_path) / "config.yaml"
        )
        assert (
            map_location is not None
        ), "map_location must be one of ('auto', 'cpu', 'cuda', 'mps')"
        device = cls._retrieve_device_str(None, map_location)
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
        inference_mode=True,
        which: str = "best",
        map_location: Literal["auto", "cpu", "cuda", "mps"] = "auto",
        **kwargs,
    ) -> Self:
        """Load a pretrained model with given name

        Args:
            pretrained_name (str): name of the pretrained model to be loaded
            inference_mode (bool, optional): whether to set the model in eval mode. Defaults to True.
            which (str, optional): which checkpoint to load. Defaults to "best".
            map_location (str, optional): device string to load the model to. Defaults to 'auto' (hardware-based).

        Returns:
            Self: loaded model
        """
        log.info(f"Loading pretrained model {pretrained_name}")
        pretrained_path = get_pretrained_model_path(pretrained_name)
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
            out["flow"]:      3d flow estimate
        """
        x = self._bg_remover(x)
        x = self._backbone(x)
        heatmaps = tuple(self._post(x))
        if self.config.compute_flow:
            flow = self._flow(x[0])
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

        if not (train_ds._sigma == val_ds._sigma == self.config.sigma):
            raise ValueError(
                "Different sigma values given for training/validation data and model!"
            )

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=logger,
            callbacks=callbacks,
            deterministic=deterministic,
            benchmark=benchmark,
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
        number_of_devices: Optional[int] = 1,
        num_workers: Optional[int] = 0,
        callbacks: Optional[Sequence[pl.callbacks.Callback]] = None,
        deterministic: Optional[bool] = True,
        benchmark: Optional[bool] = False,
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
            logger (Optional[pl.loggers.Logger], optional): logger to use. If not given, will use TensorBoard. Defaults to None.
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

        min_img_size = min(min(img.shape[:2]) for img in train_images)
        min_crop_size = int(2 ** np.floor(np.log2(min_img_size)))

        crop_size = tuple(2 * [min(train_config.crop_size, min_crop_size)])

        point_priority = 0.8 if train_config.smart_crop else 0.0
        # Build augmenters
        if augment_train:
            tr_augmenter = self.build_image_augmenter(crop_size, point_priority=point_priority)
        else:
            tr_augmenter = self.build_image_cropper(crop_size, point_priority=point_priority)
        val_augmenter = self.build_image_cropper(crop_size, point_priority=point_priority)

        # Generate datasets
        train_ds = SpotsDataset(
            train_images, train_spots, augmenter=tr_augmenter, **train_dataset_kwargs
        )
        val_ds = SpotsDataset(
            val_images, val_spots, augmenter=val_augmenter, **val_dataset_kwargs
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
            logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)
        elif logger == "wandb":
            logger = pl.loggers.WandbLogger(save_dir=save_dir)
        else:
            if logger != "none":
                log.warning(f"Logger {logger} not implemented. Using no logger.")
            logger = None

        self.fit_dataset(
            train_ds,
            val_ds,
            train_config,
            accelerator=device,
            logger=logger,
            devices=number_of_devices,
            num_workers=num_workers,
            callbacks=callbacks,
            deterministic=deterministic,
            benchmark=benchmark,
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
            assert all(
                img.ndim in (2, 3) for img in imgs
            ), f"Images must be 2D (Y,X) or 3D (Y,X,C) for {split} set"
            if self.config.in_channels > 1:
                assert all(
                    img.ndim == 3 and img.shape[-1] == self.config.in_channels
                    for img in imgs
                ), f"All images must be 2D (Y,X) for {split} set"
            assert all(
                pts.ndim == 2 and pts.shape[1] == 2 for pts in pts
            ), f"Points must be 2D (Y,X) for {split} set"
        return

    def build_image_cropper(self, crop_size: Tuple[int, int], point_priority: float = 0.):
        """Build default cropper for a dataset.

        Args:
            crop_size (Tuple[int, int]): tuple of (height, width) to randomly crop the images to.
            point_priority (float, optional): priority to sample regions containing spots when cropping. If 0, no priority is given (so all crops will be random). If 1, all crops will containg at least one spot. Defaults to 0.
        """
        cropper = AugmentationPipeline()
        cropper.add(transforms.Crop(probability=1.0, size=crop_size, point_priority=point_priority))
        return cropper

    def build_image_augmenter(
        self,
        crop_size: Optional[Tuple[int, int]] = None,
        point_priority: float = 0.,
    ) -> AugmentationPipeline:
        """Build default augmenter for training data.
           
        Args:
            crop_size (Optional[Tuple[int, int]]): if given as a tuple of (height, width), will add random cropping at the beginning of the pipeline. If None, will not crop. Defaults to None.
            point_priority (float, optional): priority to sample regions containing spots when cropping. Defaults to 0.
        """
        augmenter = (
            self.build_image_cropper(crop_size, point_priority)
            if crop_size is not None
            else AugmentationPipeline()
        )
        augmenter.add(transforms.FlipRot90(probability=0.5))
        augmenter.add(transforms.Rotation(probability=0.5, order=1))
        augmenter.add(transforms.GaussianNoise(probability=0.5, sigma=(0, 0.05)))
        augmenter.add(
            transforms.IntensityScaleShift(
                probability=0.5, scale=(0.5, 2.0), shift=(-0.2, 0.2)
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

        # ! For retrocompatibility, remove in the future
        if not states_path.exists():
            states_path = Path(path) / f"{which}.ckpt"

        checkpoint = torch.load(states_path, map_location=map_location)
        model_state = self.cleanup_state_dict_keys(checkpoint["state_dict"])
        self.load_state_dict(model_state)

        if f"prob_thresh_{which}" in thresholds.keys():
            self._prob_thresh = thresholds[f"prob_thresh_{which}"]
        else:  # ! For retrocompatibility, remove in the future
            self._prob_thresh = thresholds.get("prob_thresh", 0.5)
        if inference_mode:
            self.eval()
            return
        return

    def predict(
        self,
        img: np.ndarray,
        prob_thresh: Optional[float] = None,
        n_tiles: Tuple[int, int] = (1, 1),
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
    ) -> Tuple[np.ndarray, SimpleNamespace]:
        """Predict spots in an image.

        Args:
            img (np.ndarray): input image
            prob_thresh (Optional[float], optional): Probability threshold for peak detection. If None, will load the optimal one. Defaults to None.
            n_tiles (Tuple[int, int], optional): Number of tiles to split the image into. Defaults to (1,1).
            min_distance (int, optional): Minimum distance between spots for NMS. Defaults to 1.
            exclude_border (bool, optional): Whether to exclude spots at the border. Defaults to False.
            scale (Optional[int], optional): Scale factor to apply to the image. Defaults to None.
            subpix (bool, optional): Whether to use the stereographic flow to compute subpixel localization. If None, will deduce from the model configuration. Defaults to None.
            peak_mode (str, optional): Peak detection mode (can be either "skimage" or "fast", which is a faster custom C++ implementation). Defaults to "skimage".
            normalizer (Optional[Union[Literal["auto"], callable]], optional): Normalizer to use. If None, will use the default normalizer. Defaults to "auto" (percentile-based normalization with p_min=1, p_max=99.8).
            verbose (bool, optional): Whether to print logs and progress. Defaults to True.
            progress_bar_wrapper (Optional[callable], optional): Progress bar wrapper to use. Defaults to None.
            device (Optional[Union[torch.device, Literal["auto", "cpu", "cuda", "mps"]]], optional): computing device to use. If None, will infer from model location. If "auto", will infer from available hardware. Defaults to None.

        Returns:
            Tuple[np.ndarray, SimpleNamespace]: Tuple of (points, details). Points are the coordinates of the spots. Details is a namespace containing the spot-wise probabilities (`prob`), the heatmap (`heatmap`), the stereographic flow (`flow`), the 2D local offset vector field (`subpix`) and the spot intensities (`intens`).
        """

        if subpix is False:
            subpix_radius = -1
        elif subpix is True:
            subpix_radius = 0
        elif subpix is None:
            subpix_radius = 0 if self.config.compute_flow else -1
        else:
            subpix_radius = int(subpix)

        assert img.ndim in (2, 3), "Image must be 2D (Y,X) or 3D (Y,X,C)"

        if device is None or isinstance(device, str):
            device = self._retrieve_device_str(device)
            device = torch.device(device)
            if device is not None:
                self.to(device)


        if verbose:
            log.info(f"Will use device: {str(device)}")
            log.info(
                f"Predicting with prob_thresh = {self._prob_thresh if prob_thresh is None else prob_thresh:.3f}, min_distance = {min_distance}"
            )

        if img.ndim == 2:
            img = img[..., None]

        img = img.astype(np.float32)
        if scale is None or scale == 1:
            x = img
        else:
            if subpix_radius >= 0:
                raise NotImplementedError(
                    "Subpixel prediction is not supported yet when scale != 1."
                )
            if verbose:
                log.info(f"Scaling image by factor {scale}")
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
        pad_shape = tuple(int(d * np.ceil(s / d)) for s, d in zip(x.shape, div_by))
        if verbose:
            log.info(f"Padding to shape {pad_shape}")
        x, padding = center_pad(x, pad_shape, mode="reflect")

        if isinstance(normalizer, str) and normalizer == "auto":
            normalizer = normalize
        if normalizer is not None and callable(normalizer):
            x = normalizer(x)

        self.eval()
        # Predict without tiling
        if all(n <= 1 for n in n_tiles):
            with torch.inference_mode():
                img_t = (
                    torch.from_numpy(x).to(device).unsqueeze(0)
                )  # Add B and C dimensions
                img_t = img_t.permute(0, 3, 1, 2)
                out = self(img_t)

                y = (
                    self._sigmoid(out["heatmaps"][0].squeeze(0).squeeze(0))
                    .detach()
                    .cpu()
                    .numpy()
                )
                if subpix_radius >= 0:
                    flow = (
                        F.normalize(out["flow"], dim=1)[0]
                        .permute(1, 2, 0)
                        .detach()
                        .cpu()
                        .numpy()
                    )

            if scale is not None and scale != 1:
                y = zoom(y, (1.0 / scale, 1.0 / scale), order=1)
            if subpix_radius >= 0:
                _subpix = flow_to_vector(
                    flow,
                    sigma=self.config.sigma,
                )
                flow = center_crop(flow, img.shape[:2])
                _subpix = center_crop(_subpix, img.shape[:2])

            y = center_crop(y, img.shape[:2])

            pts = prob_to_points(
                y,
                prob_thresh=self._prob_thresh if prob_thresh is None else prob_thresh,
                exclude_border=exclude_border,
                mode=peak_mode,
                min_distance=min_distance,
            )
            probs = y[tuple(pts.astype(int).T)].tolist()

        else:  # Predict with tiling
            y = np.empty(x.shape[:2], np.float32)
            if subpix_radius >= 0:
                _subpix = np.empty(x.shape[:2] + (2,), np.float32)
            flow = np.empty(x.shape[:2] + (3,), np.float32)  # ! Check dimensions
            points = []
            probs = []
            iter_tiles = tile_iterator(
                x,
                n_tiles=n_tiles + (1,)
                if x.ndim == 3 and len(n_tiles) == 2
                else n_tiles,
                block_sizes=div_by,
                n_block_overlaps=(4, 4) if x.ndim == 2 else (4, 4, 0),
            )

            if verbose and callable(progress_bar_wrapper):
                iter_tiles = progress_bar_wrapper(iter_tiles)
            elif verbose:
                iter_tiles = tqdm(
                    iter_tiles, desc="Predicting tiles", total=np.prod(n_tiles)
                )

            for tile, s_src, s_dst in iter_tiles:
                with torch.inference_mode():
                    img_t = (
                        torch.from_numpy(tile).to(device).unsqueeze(0)
                    )  # Add B and C dimensions
                    img_t = img_t.permute(0, 3, 1, 2)
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
                    # remove global offset
                    p -= np.array([s.start for s in s_src[:2]])[None]
                    write_shape = tuple(s.stop - s.start for s in s_dst[:2])
                    p = filter_shape(p, write_shape, idxr_array=p)

                    y_tile_sub = y_tile[s_src[:2]]

                    probs += y_tile_sub[tuple(p.astype(int).T)].tolist()

                    # add global offset
                    p += np.array([s.start for s in s_dst[:2]])[None]
                    points.append(p)
                    y[s_dst[:2]] = y_tile_sub

                    # Flow
                    if subpix_radius >= 0:
                        flow_tile = (
                            F.normalize(out["flow"], dim=1)[0]
                            .permute(1, 2, 0)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        flow_tile_sub = flow_tile[s_src[:2]]
                        flow[s_dst[:2]] = flow_tile_sub

                        # Cartesian coordinates
                        subpix_tile = flow_to_vector(flow_tile, sigma=self.config.sigma)
                        subpix_tile_sub = subpix_tile[s_src[:2]]
                        _subpix[s_dst[:2]] = subpix_tile_sub

            if scale is not None and scale != 1:
                y = zoom(y, (1.0 / scale, 1.0 / scale), order=1)

            y = center_crop(y, img.shape[:2])
            if subpix_radius >= 0:
                flow = center_crop(flow, img.shape[:2])
                _subpix = center_crop(_subpix, img.shape[:2])

            points = np.concatenate(points, axis=0)

            # Remove padding
            points = points - np.array((padding[0][0], padding[1][0]))[None]

            probs = np.array(probs)
            # if scale is not None and scale != 1:
            #     points = np.round((points.astype(float) / scale)).astype(int)

            probs = filter_shape(probs, img.shape[:2], idxr_array=points)
            pts = filter_shape(points, img.shape[:2], idxr_array=points)

        if verbose:
            log.info(f"Found {len(pts)} spots")

        if subpix_radius >= 0:
            _offset = subpixel_offset(pts, _subpix, y, radius=subpix_radius)
            pts = pts + _offset
            pts = pts.clip(
                0, np.array(img.shape[:2]) - 1
            )  # FIXME: Quick fix for a corner case - should be done by subsetting the points instead similar to filter_shape
        else:
            _subpix = None
            flow = None

        # Retrieve intensity of the spots
        if subpix_radius < 0: # no need to interpolate if subpixel precision is not used
            intens = img[tuple(pts.astype(int).T)]
        else:
            try:
                intens = bilinear_interp_points(img, pts)
            except Exception as _:
                log.warn("Bilinear interpolation failed to retrive spot intensities. Will use nearest neighbour interpolation instead.")
                intens = img[tuple(pts.round().astype(int).T)]

        details = SimpleNamespace(prob=probs, heatmap=y, subpix=_subpix, flow=flow, intens=intens)
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
        log.info(
            f"Predicting with prob_thresh = {self._prob_thresh if prob_thresh is None else prob_thresh}, min_distance = {min_distance}"
        )
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
                prob_thresh=self._prob_thresh if prob_thresh is None else prob_thresh,
                exclude_border=exclude_border,
                min_distance=min_distance,
            )
            for pred in preds
        ]
        return p

    def optimize_threshold(
        self,
        val_ds: torch.utils.data.Dataset,
        cutoff_distance: int = 3,
        min_distance: int = 1,
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
                    self._sigmoid(out["heatmaps"][0].squeeze(1)).detach().cpu().numpy()
                )
                if subpix:
                    curr_flow_preds = []
                    for flow in out["flow"]:
                        curr_flow_preds += [
                            F.normalize(flow, dim=1)
                            .permute(1, 2, 0)
                            .detach()
                            .cpu()
                            .numpy()
                        ]

                for p in val_batch["pts"]:
                    val_gt_pts.append(p.numpy())

                for batch_elem in range(high_lv_hm_preds.shape[0]):
                    val_hm_preds += [high_lv_hm_preds[batch_elem]]
                    if subpix:
                        val_flow_preds += [
                            flow_to_vector(
                                curr_flow_preds[batch_elem], sigma=self.config.sigma
                            )
                        ]
                del out, imgs, val_batch

        def _metric_at_threshold(thr):
            val_pred_pts = [
                prob_to_points(
                    p,
                    prob_thresh=thr,
                    exclude_border=exclude_border,
                    min_distance=min_distance,
                )
                for p in val_hm_preds
            ]
            if subpix:
                val_pred_pts = [
                    pts + _subpix[tuple(pts.astype(int).T)]
                    for pts, _subpix in zip(val_pred_pts, val_flow_preds)
                ]
            stats = points_matching_dataset(
                val_gt_pts, val_pred_pts, cutoff_distance=cutoff_distance, by_image=True
            )
            return stats.f1

        def _grid_search(tmin, tmax):
            thr = np.linspace(tmin, tmax, niter)
            ys = tuple(
                _metric_at_threshold(t) for t in tqdm(thr, desc="optimizing threshold")
            )
            i = np.argmax(ys)
            i1, i2 = max(0, i - 1), min(i + 1, len(thr) - 1)
            return thr[i], (thr[i1], thr[i2]), ys[i]

        _, t_bounds, _ = _grid_search(*threshold_range)
        best_thr, _, best_f1 = _grid_search(*t_bounds)
        log.info(f"Best threshold: {best_thr:.3f}")
        log.info(f"Best F1-score: {best_f1:.3f}")
        self._prob_thresh = float(best_thr)
        return

    def _retrieve_device_str(
        self, device_str: Union[None, Literal["auto", "cpu", "cuda", "mps"]]
    ) -> str:
        """Retrieve the device string to use for the model.

        Args:
            device_str (Union[None, Literal["auto", "cpu", "cuda", "mps"]]): device string to use.
                If None, will use the location of the model parameters. If "auto", will infer from available hardware. Defaults to None.

        Returns:
            str: device string to use
        """
        if device_str is not None and device_str not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError(
                f"device must be one of 'auto', 'cpu', 'cuda', 'mps', got {device_str}"
            )
        if device_str is None:
            return str(next(self.parameters()).device)
        elif device_str == "auto":
            return (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
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
            return UNetBackbone(concat_mode="cat", **backbone_params)
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
