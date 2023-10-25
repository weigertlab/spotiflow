from collections import OrderedDict
from itertools import product
from csbdeep.internals.predict import tile_iterator
from pathlib import Path
from scipy.ndimage import zoom
from tqdm.auto import tqdm
from types import SimpleNamespace
from typing import Literal, Optional, Sequence, Tuple, Union


import lightning.pytorch as pl
import logging
import pydash
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml

from .backbones import ResNetBackbone, UNetBackbone
from .bg_remover import BackgroundRemover
from .config import SpotipyModelConfig, SpotipyTrainingConfig
from .post import FeaturePyramidNetwork, MultiHeadProcessor
from .trainer import SpotipyTrainingWrapper
from ..utils import (
    center_crop,
    center_pad,
    filter_shape,
    generate_datasets,
    points_matching_dataset,
    prob_to_points,
    flow_to_vector,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _subpixel_offset(
    pts: np.ndarray, subpix: np.ndarray, prob: np.ndarray, radius: int
):
    """compute offset for subpixel localization at given locations by aggregating within a radius the
    2d offset field `subpix` around each point in `pts` weighted by the probability `prob`
    """
    assert (
        pts.ndim == 2
        and pts.shape[1] == 2
        and subpix.ndim == 3
        and subpix.shape[2] == 2
        and prob.ndim == 2
    )
    n, _ = pts.shape
    _weight = np.zeros((n, 1), np.float32)
    _add = np.zeros((len(pts), 2), np.float32)
    for i, j in product(range(-radius, radius + 1), repeat=2):
        dp = np.array([[i, j]])
        p = pts + dp
        _p = tuple(p.astype(int).T)
        _w = prob[_p][:, None]
        _correct = subpix[_p] + dp
        _weight += _w
        _add += _w * _correct
    _add /= _weight
    return _add


class Spotipy(nn.Module):
    """Supervised spot detector using a multi-stage neural network as a backbone for
    feature extraction followed by resolution-dependent post-processing modules
    to allow loss computation and optimization at different resolution levels.
    """

    def __init__(self, config: Optional[SpotipyModelConfig] = None) -> None:
        """Initialize the model.

        Args:
            config (Optional[SpotipyModelConfig]): model configuration object. If None, will use the default configuration.
        """
        super().__init__()
        if config is None:
            log.info("No model config given, using default.")
            config = SpotipyModelConfig()
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
    def from_pretrained(
        cls,
        pretrained_path: str,
        inference_mode=True,
        which: str = "best",
        map_location: str = "cuda",
    ) -> None:
        """Load a pretrained model.

        Args:
            pretrained_path (str): path to the pretrained model
            inference_mode (bool, optional): whether to set the model in eval mode. Defaults to True.
            map_location (str, optional): device string to load the model to. Defaults to 'cuda'.
        """
        model_config = SpotipyModelConfig.from_config_file(
            Path(pretrained_path) / "config.yaml"
        )
        model = cls(model_config)
        model.load(
            pretrained_path,
            which=which,
            inference_mode=inference_mode,
            map_location=map_location,
        )
        return model

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

    def fit(
        self,
        training_data: Union[torch.utils.data.Dataset, Tuple],
        validation_data: Optional[Union[torch.utils.data.Dataset, Tuple]] = None,
        train_config: Optional[SpotipyTrainingConfig] = None,
        accelerator: str = "cpu",
        logger: Optional[pl.loggers.Logger] = None,
        devices: Optional[int] = 1,
        num_workers: Optional[int] = 0,
        callbacks: Optional[Sequence[pl.callbacks.Callback]] = [],
        deterministic: Optional[bool] = True,
        benchmark: Optional[bool] = False,
        **kwargs,
    ):
        """Train the model.

        Args:
            training_data (Union[torch.utils.data.Dataset, Tuple]): training dataset. If a tuple is given, it should be (I, P) where I is a list of images and P is a list of points.
            validation_data (Union[torch.utils.data.Dataset, Tuple]): validation dataset. If a tuple is given, it should be (I, P) where I is a list of images and P is a list of points.
            train_config (SpotipyTrainingConfig): training configuration
            accelerator (str): accelerator to use. Can be "cpu", "cuda", "mps" or "auto".
            logger (Optional[pl.loggers.Logger], optional): logger to use. Defaults to None.
            devices (Optional[int], optional): number of accelerating devices to use. Defaults to 1.
            num_workers (Optional[int], optional): number of workers to use for data loading. Defaults to 0.
            callbacks (Optional[Sequence[pl.callbacks.Callback]], optional): callbacks to use during training. Defaults to no callbacks.
            deterministic (Optional[bool], optional): whether to use deterministic training. Set to True for deterministic behaviour at a cost of performance. Defaults to True.
            benchmark (Optional[bool], optional): whether to use benchmarking. Set to False for deterministic behaviour at a cost of performance. Defaults to False.
            kwargs: additional arguments to pass to the SpotsDataset class. Only used if training_data is input as tuples.
        """
        if train_config is None:
            log.info("No training config given. Using default.")
            train_config = SpotipyTrainingConfig()
            log.info(f"Default training config: {train_config}")
        assert validation_data is None or type(training_data) == type(
            validation_data
        ), "Training and validation data must be of the same type"
        if isinstance(training_data, torch.utils.data.Dataset):
            train_ds, val_ds = training_data, validation_data
        else:
            if validation_data is None:
                validation_data = [None, None]
            train_ds, val_ds = generate_datasets(
                *training_data, *validation_data, **kwargs
            )

        if not training_data._sigma == validation_data._sigma == self.config.sigma:
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
        training_wrapper = SpotipyTrainingWrapper(self, train_config)
        train_dl, val_dl = training_wrapper.generate_dataloaders(
            train_ds,
            val_ds,
            num_train_samples=train_config.num_train_samples,
            num_workers=num_workers,
        )
        trainer.fit(training_wrapper, train_dl, val_dl)
        log.info("Training finished.")

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
            only_config (bool, optional): whether to log only the config (useful if re-saving only for threshold optimization). Defaults to False.
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
        subpix: Optional[Union[bool, int]] = False,
        peak_mode: Literal["skimage", "fast"] = "skimage",
        normalizer: Optional[callable] = None,
        verbose: bool = True,
        progress_bar_wrapper: Optional[callable] = None,
        device: str = "cuda",
    ) -> Tuple[np.ndarray, SimpleNamespace]:
        """Predict spots in an image.

        Args:
            img (np.ndarray): input image
            prob_thresh (Optional[float], optional): Probability threshold for peak detection. If None, will load the optimal one. Defaults to None.
            n_tiles (Tuple[int, int], optional): Number of tiles to split the image into. Defaults to (1,1).
            min_distance (int, optional): Minimum distance between spots for NMS. Defaults to 1.
            exclude_border (bool, optional): Whether to exclude spots at the border. Defaults to False.
            scale (Optional[int], optional): Scale factor to apply to the image. Defaults to None.
            subpix (bool, optional): Whether to use the stereographic flow to compute subpixel localization. Defaults to False.
            peak_mode (str, optional): Peak detection mode (can be either "skimage" or "fast", which is a faster custom C++ implementation). Defaults to "skimage".
            normalizer (Optional[callable], optional): Normalization function to apply to the image. If n_tiles is different than (1,1), then normalization is applied tile-wise. If None, no normalization is applied. Defaults to None.
            verbose (bool, optional): Whether to print logs and progress. Defaults to True.
            progress_bar_wrapper (Optional[callable], optional): Progress bar wrapper to use. Defaults to None.
            device (str, optional): Device to use for prediction. Defaults to "cuda".

        Returns:
            Tuple[np.ndarray, SimpleNamespace]: Tuple of (points, details). Points are the coordinates of the spots. Details is a namespace containing the spot-wise probabilities and the heatmap.
        """

        if subpix is False:
            subpix_radius = -1
        elif subpix is True:
            subpix_radius = 0
        else:
            subpix_radius = int(subpix)

        assert img.ndim in (2, 3), "Image must be 2D (Y,X) or 3D (Y,X,C)"
        device = torch.device(device)
        if verbose:
            log.info(
                f"Predicting with prob_thresh = {self._prob_thresh if prob_thresh is None else prob_thresh:.3f}, min_distance = {min_distance}"
            )

        if img.ndim == 2:
            img = img[..., None]

        img = img.astype(np.float32)
        if scale is None or scale == 1:
            x = img
        else:
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

        self.eval()
        # Predict without tiling
        if all(n <= 1 for n in n_tiles):
            if normalizer is not None and callable(normalizer):
                x = normalizer(x)
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
                flow = (
                    F.normalize(out["flow"], dim=1)[0]
                    .permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                )

            if scale is not None and scale != 1:
                y = zoom(y, (1.0 / scale, 1.0 / scale), order=1)
                raise NotImplementedError("flow")

            _subpix = flow_to_vector(
                flow,
                sigma=self.config.sigma,
            )
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
                # assert all(s%t == 0 for s, t in zip(tile.shape, n_tiles)), "Currently, tile shape must be divisible by n_tiles"
                if normalizer is not None and callable(normalizer):
                    tile = normalizer(tile)
                with torch.inference_mode():
                    img_t = (
                        torch.from_numpy(tile).to(device).unsqueeze(0)
                    )  # Add B and C dimensions
                    img_t = img_t.permute(0, 3, 1, 2)
                    y_tile = self._sigmoid(
                        self(img_t)["heatmaps"][0].squeeze(0).squeeze(0)
                    )
                    y_tile = y_tile.detach().cpu().numpy()

                    p = prob_to_points(
                        y_tile,
                        prob_thresh=self._prob_thresh
                        if prob_thresh is None
                        else prob_thresh,
                        exclude_border=exclude_border,
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

            if scale is not None and scale != 1:
                y = zoom(y, (1.0 / scale, 1.0 / scale), order=1)

            y = center_crop(y, img.shape[:2])

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
            _offset = _subpixel_offset(pts, _subpix, y, radius=subpix_radius)
            pts = pts + _offset

        details = SimpleNamespace(prob=probs, heatmap=y, subpix=_subpix, flow=flow)
        return pts, details

    def predict_dataset(
        self,
        ds: torch.utils.data.Dataset,
        min_distance=1,
        exclude_border=False,
        batch_size=4,
        prob_thresh=None,
        return_heatmaps=False,
        device: str = "cpu",
    ):
        """Predict spots from a torch dataset.

        Args:
            ds (torch.utils.data.Dataset): dataset to predict
            min_distance (int, optional): Minimum distance between spots for NMS. Defaults to 1.
            exclude_border (bool, optional): Whether to exclude spots at the border. Defaults to False.
            batch_size (int, optional): Batch size to use for prediction. Defaults to 4.
            prob_thresh (Optional[float], optional): Probability threshold for peak detection. If None, will load the optimal one. Defaults to None.
            return_heatmaps (bool, optional): Whether to return the heatmaps. Defaults to False.
            device (str, optional): Device to use for prediction. Defaults to 'cpu'.
        """
        preds = []
        dataloader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
        )
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
        min_distance: int = 2,
        exclude_border: bool = False,
        threshold_range: Tuple[float, float] = (0.3, 0.7),
        niter: int = 11,
        batch_size: int = 2,
        device: torch.device = torch.device("cpu"),
        subpix: bool = False,
    ) -> None:
        """Optimize the probability threshold on an annotated dataset.

        Args:
            val_ds (torch.utils.data.Dataset): dataset to optimize on
            cutoff_distance (int, optional): distance tolerance considered for points matching. Defaults to 3.
            min_distance (int, optional): Minimum distance between spots for NMS. Defaults to 1.. Defaults to 2.
            exclude_border (bool, optional): Whether to exclude spots at the border. Defaults to False.
            threshold_range (Tuple[float, float], optional): Range of thresholds to consider. Defaults to (.3, .7).
            niter (int, optional): number of iterations for both coarse- and fine-grained search. Defaults to 11.
            batch_size (int, optional): batch size to use. Defaults to 2.
            device (_type_, optional): computing device. Defaults to torch.device("cpu").
        """
        val_hm_preds = []
        val_flow_preds = []
        val_gt_pts = []
        val_dataloader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
        )
        self.eval()
        with torch.inference_mode():
            for val_batch in val_dataloader:
                imgs = val_batch["img"].to(device)

                out = self(imgs)
                high_lv_hm_preds = (
                    self._sigmoid(out["heatmaps"][0].squeeze(1)).detach().cpu().numpy()
                )
                curr_flow_preds = []
                for flow in out["flow"]:
                    curr_flow_preds += [
                        F.normalize(flow, dim=1).permute(1, 2, 0).detach().cpu().numpy()
                    ]

                for p in val_batch["heatmap_lv0"]:
                    val_gt_pts.append(
                        prob_to_points(
                            p[0].detach().cpu().numpy(),
                            prob_thresh=0.8,
                            exclude_border=exclude_border,
                            min_distance=min_distance,
                        )
                    )

                for batch_elem in range(high_lv_hm_preds.shape[0]):
                    val_hm_preds += [high_lv_hm_preds[batch_elem]]
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
