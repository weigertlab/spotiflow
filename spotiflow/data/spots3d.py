from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Sequence, Union
from typing_extensions import Self
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

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)


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
                centers.numpy(), img.shape[-3:], sigma=self._sigma, grid=self._grid,
            ).transpose((3, 0, 1, 2))
            flow = torch.from_numpy(flow).float()


        heatmap_lv0 = utils.points_to_prob3d(
            centers.numpy(), img.shape[-3:], mode=self._mode, sigma=self._sigma, grid=self._grid,
        )

        # Build target at different resolution levels
        heatmaps = [
            utils.multiscale_decimate(heatmap_lv0, ds, is_3d=True)
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

    #FIXME: duplicated code, should be gone when class labels are allowed in 3D
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
        add_class_label: bool = False,
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
        assert not add_class_label, "add_class_label not supported for 3D datasets yet."
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
            utils.read_coords_csv3d(center).astype(np.float32)
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


    def save(self, path, prefix="img_"):
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        for i, (x, y) in tqdm(
            enumerate(zip(self.images, self._centers)), desc="Saving", total=len(self)
        ):
            tifffile.imwrite(path / f"{prefix}{i:05d}.tif", x)
            pd.DataFrame(y, columns=("Z", "Y", "X")).to_csv(path / f"{prefix}{i:05d}.csv")
