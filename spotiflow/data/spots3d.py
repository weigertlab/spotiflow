from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Sequence, Union
from typing_extensions import Self
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from skimage import io
import logging
import numpy as np
import sys
import torch
import tifffile
from itertools import chain
import pandas as pd

from .spots import SpotsDataset
from .. import utils

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)


class Spots3DDataset(SpotsDataset):
    """Base spot dataset class instantiated with loaded images and centers."""
    def __getitem__(self, idx: int) -> Dict:
        img, centers = self.images[idx], self._centers[idx]
        img = torch.from_numpy(img.copy()).unsqueeze(0)  # Add B dimension
        centers = torch.from_numpy(centers.copy()).unsqueeze(0)  # Add B dimension

        assert img.ndim in (4, 5)  # Images should be in BCDWH or BDHW format
        if img.ndim == 4:
            img = img.unsqueeze(1) # Add C dimension

        img, centers = self.augmenter(img, centers)
        img, centers = img.squeeze(0), centers.squeeze(0)  # Remove B dimension

        if self._compute_flow:
            flow = utils.points_to_flow3d(
                centers.numpy(), img.shape[-3:], sigma=self._sigma
            ).transpose((3, 0, 1, 2))
            flow = torch.from_numpy(flow).float()


        heatmap_lv0 = utils.points_to_prob3d(
            centers.numpy(), img.shape[-3:], mode=self._mode, sigma=self._sigma
        )

        # Build target at different resolution levels
        heatmaps = [
            utils.multiscale_decimate(heatmap_lv0, ds)
            for ds in self._downsample_factors
        ]

        # Cast to tensor and add channel dimension
        ret_obj = {"img": img.float(), "pts": centers.float()}

        if self._compute_flow:
            ret_obj.update({"flow": flow})

        ret_obj.update(
            {
                f"heatmap_lv{lv}": torch.from_numpy(heatmap.copy()).unsqueeze(0)
                for lv, heatmap in enumerate(heatmaps)
            }
        )
        return ret_obj


    def save(self, path, prefix="img_"):
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        for i, (x, y) in tqdm(
            enumerate(zip(self.images, self._centers)), desc="Saving", total=len(self)
        ):
            tifffile.imwrite(path / f"{prefix}{i:05d}.tif", x)
            pd.DataFrame(y, columns=("Z", "Y", "X")).to_csv(path / f"{prefix}{i:05d}.csv")
