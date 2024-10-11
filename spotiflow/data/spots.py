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

from .. import utils

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)


class SpotsDataset(Dataset):
    """Base spot dataset class instantiated with loaded images and centers."""

    def __init__(
        self,
        images: Sequence[np.ndarray],
        centers: Sequence[np.ndarray],
        augmenter: Optional[Callable] = None,
        downsample_factors: Sequence[int] = (1,),
        sigma: float = 1.,
        mode: str = "max",
        compute_flow: bool = False,
        image_files: Optional[Sequence[str]] = None,
        normalizer: Union[Literal["auto"], Callable, None] = "auto",
        add_class_label: bool = True,
        grid: Optional[Sequence[int]] = None,
    ) -> Self:
        """ Constructor

        Args:
            images (Sequence[np.ndarray]): Sequence of images.
            centers (Sequence[np.ndarray]): Sequence of center coordinates.
            augmenter (Optional[Callable], optional): Augmenter function. If given, function arguments should two (first image, second spots). Defaults to None.
            downsample_factors (Sequence[int], optional): Downsample factors. Defaults to (1,).
            sigma (float, optional): Sigma of Gaussian kernel to generate heatmap. Defaults to 1.
            mode (str, optional): Mode of heatmap generation. Defaults to "max".
            compute_flow (bool, optional): Whether to compute flow from centers. Defaults to False.
            image_files (Optional[Sequence[str]], optional): Sequence of image filenames. If the dataset was not constructed from a folder, this will be None. Defaults to None.
            normalizer (Union[Literal["auto"], Callable, None], optional): Normalizer function. Defaults to "auto" (percentile-based normalization with p_min=1 and p_max=99.8).
        """
        super().__init__()

        # Build downsample factors as tuple in case input list contains integers
        self._downsample_factors = [
            t if isinstance(t, tuple) else (t, t) for t in downsample_factors
        ]

        self._centers = centers
        self._images = images

        if isinstance(normalizer, str) and normalizer == "auto":
            normalizer = utils.normalize

        if callable(normalizer):
            self._images = [normalizer(img) for img in tqdm(self.images, desc="Normalizing images")]

        if not len(centers) == len(images):
            raise ValueError("Different number of images and centers given!")

        self._compute_flow = compute_flow
        self._sigma = sigma
        self._mode = mode
        self._augmenter = (
            augmenter if augmenter is not None else lambda img, pts: (img, pts)
        )
        self._image_files = image_files
        self._n_classes = 1
        if add_class_label:
            assert all(p.shape[1] == centers[0].shape[1] for p in centers), "All center arrays should have the same number of columns!"
            if centers[0].shape[1] == 3:
                assert min(p[:, 2].min() for p in centers if p.shape[0] > 0) == 0, "Class labels should start at 0!"
                self._n_classes = max(p[:, 2].astype(int).max() for p in centers if p.shape[0] > 0) + 1
            else:
                self._centers = [np.concatenate([p, np.zeros((p.shape[0], 1))], axis=1) for p in self._centers]
        if grid is None:
            grid = (1,)*centers[0].shape[1]
        self._grid = grid
        assert len(self._grid) == centers[0].shape[1], "Grid size should have the same dimensionality as the input!"
        assert all(g > 0 for g in self._grid), "Grid size should be positive!"
        assert all(g == 1 or g%2 == 0 for g in self._grid), "Grid elements should be either 1 or an odd integer > 1!"

    @classmethod
    def from_folder(
        cls,
        path: Union[Path, str],
        augmenter: Optional[Callable] = None,
        downsample_factors: Sequence[int] = (1,),
        sigma: float = 1.0,
        image_extensions: Sequence[str] = ("tif", "tiff", "png", "jpg", "jpeg"),
        mode: str = "max",
        max_files: Optional[int] = None,
        compute_flow: bool = False,
        normalizer: Optional[Union[Callable, Literal["auto"]]] = "auto",
        random_state: Optional[int] = None,
        add_class_label: bool = True,
        grid: Optional[Sequence[int]] = None,
    ) -> Self:
        """Build dataset from folder. Images and centers are loaded from disk and normalized.

        Args:
            path (Union[Path, str]): Path to folder containing images (with given extensions) and centers.
            augmenter (Callable): Augmenter function.
            downsample_factors (Sequence[int], optional): Downsample factors. Defaults to (1,).
            sigma (float, optional): Sigma of Gaussian kernel to generate heatmap. Defaults to 1.
            image_extensions (Sequence[str], optional): Image extensions to look for in images. Defaults to ("tif", "tiff", "png", "jpg", "jpeg").
            mode (str, optional): Mode of heatmap generation. Defaults to "max".
            max_files (Optional[int], optional): Maximum number of files to load. Defaults to None (all of them).
            compute_flow (bool, optional): Whether to compute flow from centers. Defaults to False.
            normalizer (Optional[Union[Callable, Literal["auto"]]], optional): Normalizer function. Defaults to "auto" (percentile-based normalization with p_min=1 and p_max=99.8).
            random_state (Optional[int], optional): Random state used when shuffling file names when "max_files" is not None. Defaults to None.
        
        Returns:
            Self: Dataset instance.
        """
        if isinstance(path, str):
            path = Path(path)
        image_files = sorted(path.glob("*.tif"))
        center_files = sorted(path.glob("*.csv"))

        image_files = sorted(
            tuple(chain(*tuple(path.glob(f"*.{ext}") for ext in image_extensions)))
        )

        if max_files is not None:
            rng = np.random.default_rng(
                random_state if random_state is not None else 42
            )
            idx = np.arange(len(image_files))
            rng.shuffle(idx)
            image_files = [image_files[i] for i in idx[:max_files]]
            center_files = [center_files[i] for i in idx[:max_files]]

        if not len(image_files) == len(center_files):
            raise ValueError(
                f"Different number of images and centers found! {len(image_files)} images, {len(center_files)} centers."
            )


        images = [io.imread(img) for img in tqdm(image_files, desc="Loading images")]

        centers = [
            utils.read_coords_csv(center, add_class_column=add_class_label).astype(np.float32)
            for center in tqdm(center_files, desc="Loading centers")
        ]


        return cls(
            images=images,
            centers=centers,
            augmenter=augmenter,
            downsample_factors=downsample_factors,
            sigma=sigma,
            mode=mode,
            compute_flow=compute_flow,
            image_files=image_files,
            normalizer=normalizer,
            add_class_label=add_class_label,
            grid=grid,
        )

    def __len__(self) -> int:
        return len(self._centers)

    def __getitem__(self, idx: int) -> Dict:
        img, centers = self.images[idx], self._centers[idx]

        img = torch.from_numpy(img.copy()).unsqueeze(0)  # Add B dimension
        centers = torch.from_numpy(centers.copy()).unsqueeze(0)  # Add B dimension

        # rgb/multichannel images should be in channel last format
        if img.ndim == 4:
            img = img.permute(0, 3, 1, 2)
        else:
            img = img.unsqueeze(1)

        img, centers = self.augmenter(img, centers)
        img, centers = img.squeeze(0), centers.squeeze(0)  # Remove B dimension

        if self._compute_flow:
            for cl in range(self._n_classes):
                curr_centers = centers[centers[:,2]==cl][:,:2]
                curr_flow = utils.points_to_flow(
                    curr_centers.numpy(), img.shape[-2:], sigma=self._sigma
                ).transpose((2, 0, 1))
                curr_flow = torch.from_numpy(curr_flow).float() # shape is (3,H,W)
                if cl == 0:
                    flow = curr_flow.clone()
                else:
                    flow = torch.cat([flow, curr_flow], dim=0) # shape will be (3*n_classes, H, W)
                del curr_flow

        for cl in range(self._n_classes):
            curr_centers = centers[centers[:,2]==cl][:,:2]
            curr_heatmap = utils.points_to_prob(
                curr_centers.numpy(), img.shape[-2:], mode=self._mode, sigma=self._sigma
            )[None, ...]
            if cl == 0:
                heatmap_lv0 = curr_heatmap.copy()
                del curr_heatmap
            else:
                heatmap_lv0 = np.concatenate([heatmap_lv0, curr_heatmap], axis=0) # shape will be (n_classes,H,W)
                del curr_heatmap

        # Build target at different resolution levels
        heatmaps = [
            utils.multiscale_decimate(heatmap_lv0, ds)
            for ds in self._downsample_factors
        ]

        # Cast to float
        ret_obj = {"img": img.float(), "pts": centers.float()}

        if self._compute_flow:
            ret_obj.update({"flow": flow})

        ret_obj.update(
            {
                f"heatmap_lv{lv}": torch.from_numpy(heatmap.copy())
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
    def images(self) -> Sequence[np.ndarray]:
        """Return images in dataset.

        Returns:
            Sequence[np.ndarray]: Sequence of images.
        """
        return self._images
    
    @property
    def n_classes(self) -> int:
        """Return number of classes in the dataset.

        Returns:
            int: number of spot classes.
        """
        return self._n_classes

    @property
    def image_files(self) -> Sequence[str]:
        """Return image filenames with the same order as in the dataset.

        Returns:
            Union[Sequence[str], None]: Sequence of image filenames. If the dataset was not constructed from a folder, this will be None.
        """
        return self._image_files

    def save(self, path, prefix="img_"):
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        for i, (x, y) in tqdm(
            enumerate(zip(self.images, self._centers)), desc="Saving", total=len(self)
        ):
            tifffile.imwrite(path / f"{prefix}{i:05d}.tif", x)
            pd.DataFrame(y, columns=("Y", "X")).to_csv(path / f"{prefix}{i:05d}.csv")


def collate_spots(batch):
    """custom collate that doesnt stack points (as they might have different length)"""

    keys_to_ignore = ("pts",)
    batch_new = dict(
        (
            k,
            torch.stack([x[k] for x in batch], dim=0),
        )
        for k, v in batch[0].items()
        if k not in keys_to_ignore
    )

    for k in keys_to_ignore:
        batch_new[k] = [x[k] for x in batch]
    return batch_new
