import pathlib
from abc import ABC
from typing import List, Sequence
from torch.utils.data import Dataset
from tormenter.transforms import Crop
from tqdm.auto import tqdm

import augmend
import logging
import numpy as np
import sys
import torch
import tifffile 
import pandas as pd

from .. import utils

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC):
    """Abstract base class"""

    def __init__(
        self,
        images: Sequence[np.ndarray],
        centers: Sequence[np.ndarray],
        downsample_factors: List[int] = [1],
        sigma: float = 1,
        mode: str = "max",
        augment_probability: float = 1,
        use_gpu: bool = False,
        size=(256, 256),
        should_center_crop=False
    ) -> None:
        super().__init__()

        # Build downsample factors as tuple in case input list contains integers
        self._downsample_factors = [
            t if isinstance(t, tuple) else (t, t) for t in downsample_factors
        ]

        self._centers = centers
        self._images = images

        if not len(centers) == len(images):
            raise ValueError(f"Different number of images and centers given!")

        self._sigma = sigma
        self._mode = mode
        self._augmenter = self._build_augmenter(augment_probability, use_gpu=use_gpu)
        
        self._size = size
        self._cropper = None
        if not should_center_crop and size is not None:
            self._cropper = Crop(probability=1, size=size)
            # self._cropper.add([augmend.RandomCrop(axis=(-2, -1), size=size), augmend.RandomCrop(axis=(-2, -1), size=size)])
        elif size is not None: # For compat with the augmend object
            self._cropper = lambda lst: [utils.center_crop(img=item, size=size) for item in lst]
        self._cache = {}
        self.crop_config = {"size": size, "center": should_center_crop}

    def __len__(self):
        return len(self._centers)

    def __getitem__(self, idx):
        # Try to get cached image
        # if (cached_item := self._retrieve_from_cache(idx)) is not None:
        #     img, heatmap_lv0 = cached_item["img"], cached_item["heatmap_lv0"]
        # else:
        img, centers = self._images[idx], self._centers[idx]
        img_size = img.shape

        img = torch.from_numpy(img.copy()).unsqueeze(0) # Add B dimension
        centers = torch.from_numpy(centers.copy()).unsqueeze(0) # Add B dimension
        
        if self._cropper is not None:
            img_size = self._size
            img, centers = self._cropper(img, centers)
        # Apply augmentations
        if self.augmenter is not None:
            img, centers = self.augmenter(img, centers)

        heatmap_lv0 = utils.points_to_prob(
            centers[0].numpy(), img_size, mode=self._mode, sigma=self._sigma
        )

        # Build target at different resolution levels
        heatmaps = [
            utils.multiscale_decimate(heatmap_lv0, ds)
            for ds in self._downsample_factors
        ]

        # Cast to tensor and add channel dimension
        try:
            ret_obj = {"img": img}
        except Exception as e:
            print(idx, img.shape, img.dtype)
            raise e
        ret_obj.update(
            {
                f"heatmap_lv{lv}": torch.from_numpy(heatmap.copy()).unsqueeze(0)
                for lv, heatmap in enumerate(heatmaps)
            }
        )
        return ret_obj

    def _build_augmenter(self, augment_probability, use_gpu=False):
        pass

    @property
    def augmenter(self):
        return self._augmenter

    # def _add_to_cache(self, idx, img, heatmap):
    #     self._cache[idx] = {}
    #     self._cache[idx]["img"] = img
    #     self._cache[idx]["heatmap_lv0"] = heatmap
    #     return

    
    def get_centers(self):
        return self._centers # !
        crop_size, center_crop = self.crop_config["size"], self.crop_config["center"]
        crop_y, crop_x = crop_size
        start_pts = [utils.center_crop(im, size=crop_size, return_indices=True) for im in self._images]
        start_ys, start_xs = [t[0] for t in start_pts], [t[1] for t in start_pts]
        if not center_crop and (any(start_ys != 0) or any(start_xs != 0)):
            log.warning("Crop required but center crop not specified. Returning all centers in the image. This makes metrics computed from these centers invalid.")
            return self._centers
        ret_centers = [None]*len(self._centers)
        for idx, curr_centers in enumerate(self._centers):
            ret_centers[idx] = np.array([(y-start_ys[idx], x-start_xs[idx]) for y, x in curr_centers if y >= start_ys[idx] and y < start_ys[idx]+crop_y and x >= start_xs[idx] and x < start_xs[idx]+crop_x])
        return ret_centers


    # def _retrieve_from_cache(self, idx):
    #     return self._cache.get(idx, None)


    def save(self, path, prefix='img_'):
        path = pathlib.Path(path)
        path.mkdir(exist_ok=True, parents=True)
        for i, (x, y) in tqdm(enumerate(zip(self._images, self._centers)), desc='Saving', total=len(self)):
            tifffile.imwrite(path/f'{prefix}{i:05d}.tif', x)
            pd.DataFrame(y, columns=('Y','X')).to_csv(path/f'{prefix}{i:05d}.csv')

