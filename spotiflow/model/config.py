import abc
import argparse
import json
import logging
import sys
from numbers import Number
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import yaml

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

class SpotiflowConfig(argparse.Namespace, abc.ABC):
    def __init__(self):
        self.is_valid()

    @classmethod
    def from_config_file(cls, f: Union[str, Path]):
        if isinstance(f, str):
            f = Path(f)
        assert f.is_file(), f"Config file {f} does not exist."
        if f.suffix == ".json":
            with open(f, "r") as fp:
                loaded_dct = json.load(fp)
        elif f.suffix in {".yml", ".yaml"}:
            with open(f, "r") as fp:
                loaded_dct = yaml.safe_load(fp)
        else:
            raise ValueError(f"Config file {f} must be either a JSON or YAML file.")
        if "downsample_factors" in loaded_dct.keys():
            loaded_dct["downsample_factors"] = tuple(
                tuple(ft) for ft in loaded_dct["downsample_factors"]
            )
        if "kernel_sizes" in loaded_dct.keys():
            loaded_dct["kernel_sizes"] = tuple(
                tuple(kt) for kt in loaded_dct["kernel_sizes"]
            )
        config = cls(**loaded_dct)
        return config

    def save(self, f: Union[str, Path]) -> None:
        if isinstance(f, str):
            f = Path(f)
        cfg_dict = vars(self)

        for k, v in cfg_dict.items():
            if isinstance(v, Path):
                cfg_dict[k] = str(v)

        if f.suffix == ".json":
            with open(f, "w") as fp:
                json.dump(cfg_dict, fp, indent=4)
        elif f.suffix in {".yml", ".yaml"}:
            with open(f, "w") as fp:
                yaml.safe_dump(cfg_dict, fp, indent=4)
        else:
            raise ValueError(f"Config file {f} must be either a JSON or YAML file.")
        return

    def __str__(self):
        pre = f"{self.__class__.__name__}(\n"
        post = "\n)"
        return (
            pre
            + "\n".join(
                [
                    f"\t{att}={val}"
                    for att, val in sorted(vars(self).items(), key=lambda x: x[0])
                ]
            )
            + post
        )

    @abc.abstractmethod
    def is_valid(self):
        pass


class SpotiflowModelConfig(SpotiflowConfig):
    def __init__(
        self,
        backbone: Literal["resnet", "unet", "unet_res"] = "unet",
        in_channels: int = 1,
        out_channels: int = 1,
        initial_fmaps: int = 32,
        fmap_inc_factor: Number = 2,
        n_convs_per_level: int = 3,
        levels: int = 4,
        downsample_factor: int = 2,
        kernel_size: int = 3,
        padding: Union[int, str] = "same",
        mode: Literal["direct", "fpn", "slim"] = "slim",
        background_remover: bool = False,
        compute_flow: bool = True,
        batch_norm: bool = True,
        downsample_factors: Optional[Tuple[Tuple[int, int]]] = None,
        kernel_sizes: Optional[Tuple[Tuple[int, int]]] = None,
        dropout: float = 0.0,
        sigma: Number = 1.0,
        is_3d: bool = False,
        grid: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        **kwargs,
    ):
        self.backbone = backbone
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_fmaps = initial_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.n_convs_per_level = n_convs_per_level
        if downsample_factors is None:
            self.downsample_factors = tuple(
                (downsample_factor,)*2 if not is_3d else (downsample_factor,)*3 for _ in range(levels)
            )
        else:
            log.debug(
                "Using downsample_factors argument. downsample_factor will be ignored."
            )
            self.downsample_factors = downsample_factors
        if kernel_sizes is None:
            self.kernel_sizes = tuple(
                (kernel_size,)*2 if not is_3d else (kernel_size,)*3
                for _ in range(n_convs_per_level)
            )
        else:
            log.debug("Using kernel_sizes argument. kernel_size will be ignored.")
            self.kernel_sizes = kernel_sizes

        if padding == "same":
            self.padding = self.kernel_sizes[0][0] // 2
        else:
            self.padding = padding

        self.levels = levels
        self.mode = mode
        self.background_remover = bool(background_remover)
        self.compute_flow = bool(compute_flow)
        self.batch_norm = bool(batch_norm)
        self.dropout = dropout
        self.sigma = sigma
        self.downsample_factor = downsample_factor
        self.is_3d = bool(is_3d)
        if isinstance(grid, int):
            self.grid = (grid,)*(3 if self.is_3d else 2)
        elif not isinstance(grid, tuple):
            self.grid = tuple(grid)
        else:
            self.grid = grid
        
        if len(self.grid) == 3 and not self.is_3d:
            self.grid = self.grid[:2]
        if any(s != 1 for s in self.grid) and not self.is_3d:
            log.warning("Grid is only used in 3D mode. Ignoring grid argument.")
        super().__init__()

    def is_valid(self):
        assert self.backbone in {
            "resnet",
            "unet",
            "unet_res"
        }, "backbone must be either 'resnet', 'unet', or 'unet_res"
        assert (
            isinstance(self.in_channels, int) and self.in_channels > 0
        ), "in_channels must be greater than 0"
        assert (
            isinstance(self.out_channels, int) and self.out_channels == 1
        ), "out_channels must be equal to 1 (multi-channel output not supported yet)"
        assert (
            isinstance(self.initial_fmaps, int) and self.initial_fmaps > 0
        ), "initial_fmaps must be greater than 0"
        assert (
            isinstance(self.n_convs_per_level, int) and self.n_convs_per_level > 0
        ), "n_convs_per_level must be greater than 0"
        assert all(
            isinstance(factor, tuple) and (len(factor) == 2 if not self.is_3d else len(factor) == 3)
            for factor in self.downsample_factors
        ), "downsample_factors must be a tuple of tuples of length 2"
        assert all(
            isinstance(f, int) and f > 0
            for factor in self.downsample_factors
            for f in factor
        ), "downsample_factors must be a tuple of tuples of integers"
        assert (
            len(self.kernel_sizes) == self.n_convs_per_level
        ), "kernel_sizes must have length equal to n_convs_per_level"
        assert all(
            isinstance(ksize, tuple) and len(ksize) == (2 if not self.is_3d else 3) for ksize in self.kernel_sizes
        ), "kernel_sizes must be a tuple of tuples of length 2"
        assert all(
            isinstance(k, int) and k > 0 for ksize in self.kernel_sizes for k in ksize
        ), "kernel_sizes must be a tuple of tuples of integers"
        assert isinstance(self.padding, int) or self.padding in {
            "same",
            "valid",
        }, "padding must be either 'same' or 'valid'"
        assert (
            isinstance(self.padding, str) or self.padding >= 0
        ), "padding must be greater than or equal to 0"
        assert self.levels > 0, "levels must be greater than 0"
        assert self.mode in {
            "direct",
            "fpn",
            "slim",
        }, "mode must be either 'direct', 'fpn', or 'slim'"
        assert 0.0 <= self.dropout <= 1.0, "dropout must be between 0 and 1"
        assert (
            isinstance(self.sigma, Number) and self.sigma >= 0
        ), "sigma must be a number >= 0."
        assert (
            isinstance(self.downsample_factor, int) and self.downsample_factor > 0
        ), "downsample_factor must be a positive integer"
        assert (
            all(isinstance(s, int) and s > 0 and (s == 1 or s % 2 == 0) for s in self.grid)
        ), "grid must be a tuple containing only 1 or even positive integers"
        assert len(np.unique(self.grid)) == 1, "grid must currently be isotropic (all dimensions must be equal)"


class SpotiflowTrainingConfig(SpotiflowConfig):
    def __init__(
        self,
        crop_size: Union[int, Tuple[int, int], Tuple[int, int, int]] = 512,
        smart_crop: bool = False,
        heatmap_loss_f: str = "bce",
        flow_loss_f: str = "l1",
        loss_levels: Optional[int] = None,
        num_train_samples: Optional[int] = None,
        pos_weight: Number = 10.0,
        lr: float = 3e-4,
        optimizer: str = "adamw",
        batch_size: int = 4,
        lr_reduce_patience: int = 10,
        num_epochs: int = 200,
        finetuned_from: Optional[str] = None,
        early_stopping_patience: int = 0,
        crop_size_depth: int = 32,
        **kwargs,
    ):
        self.crop_size = crop_size
        self.smart_crop = bool(smart_crop)
        self.heatmap_loss_f = heatmap_loss_f

        # FIXME DEPRECATED. Remove in future versions
        if (loss_f := kwargs.get("loss_f", None)) is not None:
            self.heatmap_loss_f = loss_f

        self.flow_loss_f = flow_loss_f
        self.loss_levels = loss_levels
        self.pos_weight = pos_weight
        self.lr = lr
        self.lr_reduce_patience = lr_reduce_patience
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_train_samples = num_train_samples
        self.finetuned_from = finetuned_from
        self.early_stopping_patience = early_stopping_patience
        self.crop_size_depth = crop_size_depth # FIXME: should be done w crop_size argument
        super().__init__()

    def is_valid(self):
        assert (
            isinstance(self.crop_size, int) and self.crop_size > 0
        ), "crop_size must be an integer > 0."
        assert self.heatmap_loss_f in {
            "bce",
            "mse",
            "smoothl1",
            "adawing",
        }, "heatmap_loss_f must be either 'bce', 'mse', 'smoothl1', or 'adawing'"
        assert self.flow_loss_f in {
            "l1",
        }, "flow_loss_f must be 'l1'"
        assert (
            isinstance(self.pos_weight, Number) and self.pos_weight > 0
        ), "pos_weight must be a number greater than 0."
        assert (
            isinstance(self.lr, float) and self.lr > 0
        ), "lr must be a floating point number greater than 0."
        assert self.optimizer in {"adamw"}, "optimizer must be 'adamw'"
        assert (
            isinstance(self.batch_size, int) and self.batch_size > 0
        ), "batch_size must be an integer greater than 0."
        assert (
            isinstance(self.num_epochs, int) and self.num_epochs >= 0
        ), "num_epochs must be a positive integer or 0"
        assert (
            isinstance(self.early_stopping_patience, int) and self.early_stopping_patience >= 0
        ), "early_stopping_patience must be >= 0"
        assert (
            isinstance(self.crop_size_depth, int) and self.crop_size_depth > 0
        ), "crop_size_depth must be an integer > 0."
