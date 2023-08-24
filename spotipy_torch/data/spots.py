from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union
from torch.utils.data import Dataset
from tqdm.auto import tqdm

import logging
import numpy as np
import sys
import torch
import tifffile 
import pandas as pd

from .. import utils

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)


class SpotsDataset(Dataset):
    """Base spot dataset class instantiated with loaded images and centers."""

    def __init__(
        self,
        images: Sequence[np.ndarray],
        centers: Sequence[np.ndarray],
        augmenter: Union[Callable, None],
        downsample_factors: Sequence[int] = (1,),
        sigma: float = 1.,
        mode: str = "max",
        image_files: Optional[Sequence[str]] = None,
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
        self._augmenter = augmenter if augmenter is not None else lambda img, pts: (img, pts)
        self._image_files = image_files

    @classmethod
    def from_folder(cls,
                    path: Union[Path, str],
                    augmenter: Union[Callable, None],
                    downsample_factors: Sequence[int] = (1,),
                    sigma: float = 1.,
                    mode: str = "max",
                    normalizer: Callable = utils.normalize) -> None:
        """Build dataset from folder. Images and centers are loaded from disk and normalized.

        Args:
            path (Union[Path, str]): Path to folder containing images and centers.
            augmenter (Callable): Augmenter function.
            downsample_factors (Sequence[int], optional): Downsample factors. Defaults to (1,).
            sigma (float, optional): Sigma of Gaussian kernel to generate heatmap. Defaults to 1.
            mode (str, optional): Mode of heatmap generation. Defaults to "max".
            normalizer (Callable, optional): Normalizer function. Defaults to percentile normalization with p=(1, 99.8).
        """
        if isinstance(path, str):
            path = Path(path)
        image_files = sorted(path.glob("*.tif"))
        center_files = sorted(path.glob("*.csv"))
        image_files = sorted([str(path/f) for f in path.iterdir() if f.suffix in {".tif", ".tiff"}])
        center_files = sorted([str(path/f) for f in path.iterdir() if f.suffix == ".csv"])


        images = [normalizer(tifffile.imread(img)) for img in image_files]
        centers = [utils.read_coords_csv(center).astype(np.int32) for center in center_files]

        return cls(
            images=images,
            centers=centers,
            augmenter=augmenter,
            downsample_factors=downsample_factors,
            sigma=sigma,
            mode=mode,
            image_files=image_files,
        )

    def __len__(self) -> int:
        return len(self._centers)

    def __getitem__(self, idx: int) -> Dict:
        img, centers = self._images[idx], self._centers[idx]
        img = torch.from_numpy(img.copy()).unsqueeze(0) # Add B dimension
        centers = torch.from_numpy(centers.copy()).unsqueeze(0) # Add B dimension

        img, centers = self.augmenter(img, centers)
        img, centers = img.squeeze(0), centers.squeeze(0) # Remove B dimension

        heatmap_lv0 = utils.points_to_prob( 
            centers.numpy(), img.shape, mode=self._mode, sigma=self._sigma
        )

        # Build target at different resolution levels
        heatmaps = [
            utils.multiscale_decimate(heatmap_lv0, ds)
            for ds in self._downsample_factors
        ]

        # Cast to tensor and add channel dimension
        ret_obj = {"img": img.unsqueeze(0)}
        ret_obj.update(
            {
                f"heatmap_lv{lv}": torch.from_numpy(heatmap.copy()).unsqueeze(0)
                for lv, heatmap in enumerate(heatmaps)
            }
        )
        return ret_obj


    @property
    def augmenter(self) -> Callable:
        """Return augmenter function.

        Returns:
            Callable: Augmenter function. It should take image and centers as input.
        """
        return self._augmenter

    @property
    def centers(self) -> Sequence[np.ndarray]:
        """Return centers of spots in dataset.

        Returns:
            Sequence[np.ndarray]: Sequence of center coordinates.
        """
        return self._centers

    @property
    def image_files(self) -> Sequence[str]:
        """Return image filenames with the same order as in the dataset.

        Returns:
            Union[Sequence[str], None]: Sequence of image filenames. If the dataset was not constructed from a folder, this will be None.
        """
        return self._image_files
    

    def save(self, path, prefix='img_'):
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        for i, (x, y) in tqdm(enumerate(zip(self._images, self._centers)), desc='Saving', total=len(self)):
            tifffile.imwrite(path/f'{prefix}{i:05d}.tif', x)
            pd.DataFrame(y, columns=('Y','X')).to_csv(path/f'{prefix}{i:05d}.csv')
