import logging
import os
import warnings
from itertools import product
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import wandb
from csbdeep.utils import normalize_mi_ma


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def read_coords_csv(fname: str) -> np.ndarray:
    """parses a csv file and returns correctly ordered points array"""
    try:
        df = pd.read_csv(fname)
    except pd.errors.EmptyDataError:
        return np.zeros((0, 2), dtype=np.int)

    df = df.rename(columns=str.lower)
    cols = set(df.columns)

    col_candidates = (("axis-0", "axis-1"), ("y", "x"), ("Y", "X"))
    points = None
    for possible_columns in col_candidates:
        if cols.issuperset(set(possible_columns)):
            points = df[list(possible_columns)].to_numpy()
            break

    if points is None:
        raise ValueError(f"could not get points from csv file {fname}")

    return points


def filter_shape(points, shape, idxr_array=None, return_mask=False) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """returns all values in "points" that are inside the shape as given by the indexer array
    if the indexer array is None, then the array to be filtered itself is used
    """
    if idxr_array is None:
        idxr_array = points.copy()
    assert idxr_array.ndim == 2 and idxr_array.shape[1] == 2
    idx = np.all(np.logical_and(idxr_array >= 0, idxr_array < np.array(shape)), axis=1)
    if return_mask:
        return points[idx], idx
    return points[idx]


def multiscale_decimate(y, decimate=(4, 4), sigma=1) -> np.ndarray:
    if decimate == (1, 1):
        return y
    assert y.ndim == len(decimate)
    from skimage.measure import block_reduce

    y = block_reduce(y, decimate, np.max)
    y = 2 * np.pi * sigma**2 * ndi.gaussian_filter(y, sigma)
    y = np.clip(y, 0, 1)
    return y


def center_pad(x, shape, mode="reflect") -> Tuple[np.ndarray, Sequence[Tuple[int, int]]]:
    """pads x to shape , inverse of center_crop"""
    if x.shape == shape:
        return x, tuple((0, 0) for _ in x.shape)
    if not all([s1 <= s2 for s1, s2 in zip(x.shape, shape)]):
        raise ValueError(f"shape of x {x.shape} is larger than final shape {shape}")
    diff = np.array(shape) - np.array(x.shape)
    pads = tuple(
        (int(np.ceil(d / 2)), d - int(np.ceil(d / 2))) if d > 0 else (0, 0)
        for d in diff
    )
    return np.pad(x, pads, mode=mode), pads


def center_crop(x, shape) -> np.ndarray:
    """crops x to shape, inverse of center_pad

    y = center_pad(x,shape)
    z = center_crop(y,x.shape)
    np.allclose(x,z)
    """
    if x.shape == shape:
        return x
    if not all([s1 >= s2 for s1, s2 in zip(x.shape, shape)]):
        raise ValueError(f"shape of x {x.shape} is smaller than final shape {shape}")
    diff = np.array(x.shape[: len(shape)]) - np.array(shape)
    ss = tuple(
        slice(int(np.ceil(d / 2)), s - d + int(np.ceil(d / 2)))
        if d > 0
        else slice(None)
        for d, s in zip(diff, x.shape)
    )
    return x[ss]


def str2bool(v) -> bool:
    if v.lower() in ("yes", "true", "True", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "False", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def str2scalar(dtype) -> callable:
    def _f(v):
        if v.lower() == "none":
            return None
        else:
            return dtype(v)

    return _f


def normalize(
    x: np.ndarray, pmin=1, pmax=99.8, subsample: int = 1, clip=False, ignore_val=None
) -> np.ndarray:
    """
    normalizes a 2d image with the additional option to ignore a value
    """

    # create subsampled version to compute percentiles
    ss_sample = tuple(
        slice(None, None, subsample) if s > 42 * subsample else slice(None, None)
        for s in x.shape
    )

    y = x[ss_sample]

    if ignore_val is not None:
        mask = y != ignore_val
    else:
        mask = np.ones(y.shape, dtype=bool)

    if not np.any(mask):
        return normalize_mi_ma(x, ignore_val, ignore_val, clip=clip)

    mi, ma = np.percentile(y[mask], (pmin, pmax))
    return normalize_mi_ma(x, mi, ma, clip=clip)


def normalize_fast2d(
    x,
    pmin=1,
    pmax=99.8,
    dst_range=(0, 1.0),
    clip=False,
    sub=4,
    blocksize=None,
    order=1,
    ignore_val=None,
) -> np.ndarray:
    """
    normalizes a 2d image
    if blocksize is not None (e.g. 512), computes adaptive/blockwise percentiles
    """
    assert x.ndim == 2

    out_slice = slice(None), slice(None)

    if blocksize is None:
        x_sub = x[::sub, ::sub]
        if ignore_val is not None:
            x_sub = x_sub[x_sub != ignore_val]
        mi, ma = np.percentile(x_sub, (pmin, pmax))  # .astype(x.dtype)
        print(f"normalizing_fast with mi = {mi:.2f}, ma = {ma:.2f}")
    else:
        from csbdeep.internals.predict import tile_iterator_1d

        try:
            import cv2
        except ImportError:
            raise ImportError(
                "normalize_adaptive() needs opencv, which is missing. Please install it via 'pip install opencv-python'"
            )

        if np.isscalar(blocksize):
            blocksize = (blocksize,) * 2

        if not all(s % b == 0 for s, b in zip(x.shape, blocksize)):
            warnings.warn(
                f"image size {x.shape} not divisible by blocksize {blocksize}"
            )
            pads = tuple(b - s % b for b, s in zip(blocksize, x.shape))
            out_slice = tuple(slice(0, s) for s in x.shape)
            print(f"padding with {pads}")
            x = np.pad(x, tuple((0, p) for p in pads), mode="reflect")

        n_tiles = tuple(max(1, s // b) for s, b in zip(x.shape, blocksize))

        print(f"normalizing_fast adaptively with {n_tiles} tiles and order {order}")
        mi, ma = np.zeros(n_tiles, x.dtype), np.zeros(n_tiles, x.dtype)

        kwargs = dict(block_size=1, n_block_overlap=0, guarantee="n_tiles")

        for i, (itile, is_src, is_dst) in enumerate(
            tile_iterator_1d(x, axis=0, n_tiles=n_tiles[0], **kwargs)
        ):
            for j, (tile, s_src, s_dst) in enumerate(
                tile_iterator_1d(itile, axis=1, n_tiles=n_tiles[1], **kwargs)
            ):
                x_sub = tile[::sub, ::sub]
                if ignore_val is not None:
                    x_sub = x_sub[x_sub != ignore_val]
                    x_sub = np.array(0) if len(x_sub) == 0 else x_sub
                mi[i, j], ma[i, j] = np.percentile(x_sub, (pmin, pmax)).astype(x.dtype)

        interpolations = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR}

        mi = cv2.resize(mi, x.shape[::-1], interpolation=interpolations[order])
        ma = cv2.resize(ma, x.shape[::-1], interpolation=interpolations[order])

    x = x.astype(np.float32)
    x -= mi
    x *= dst_range[1] - dst_range[0]
    x /= ma - mi + 1e-20
    x = x[out_slice]

    x += dst_range[0]

    if clip:
        x = np.clip(x, *dst_range)
    return x


def initialize_wandb(options, train_dataset, val_dataset, silent=True) -> None:
    if options.get("skip_logging"):
        log.info("Run won't be logged to wandb")
        return None
    else:
        log.info(f"Initializing wandb project for user '{options['wandb_user']}'")
    if silent:
        os.environ["WANDB_SILENT"] = "true"
    try:
        wandb.init(
            project=options["wandb_project"],
            entity=options["wandb_user"],
            name=options["run_name"],
            config=options,
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(
            {"n_train_samples": len(train_dataset), "n_val_samples": len(val_dataset)}
        )
        log.info("wandb initialized successfully")
    except KeyError as ke:
        log.warn(f"Skipping logging to wandb due to missing options: {ke}")
        return None
    return None


def write_coords_csv(pts: np.ndarray, fname: Path) -> None:
    """writes points to csv file"""
    df = pd.DataFrame(pts, columns=["y", "x"])
    df.to_csv(fname, index=False)
    return

def remove_device_id_from_device_str(device_str):
    return device_str.split(":")[0].strip()

def get_data(path: Union[Path, str], normalize: bool=True, include_test: bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get data from a given path. The path should contain a 'train' and 'val' folder.

    Args:
        path (Union[Path, str]): Path to the data.
        normalize (bool, optional): Whether to normalize the data. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A 4-length tuple of arrays corresponding to the training images, training spots, validation images and validation spots.
    """
    from ..data import SpotsDataset

    if isinstance(path, str):
        path = Path(path)
    
    assert path.exists(), f"Given data path {path} does not exist!"

    train_path = path/"train"
    val_path = path/"val"
    test_path = path/"test"
    assert (train_path).exists(), f"Given data path {path} does not contain a 'train' folder!"
    assert (val_path).exists(), f"Given data path {path} does not contain a 'val' folder!"
    if include_test:
        assert (test_path).exists(), f"Given data path {path} does not contain a 'test' folder!"

    test_ds = None
    if normalize:
        tr_ds = SpotsDataset.from_folder(train_path)
        val_ds = SpotsDataset.from_folder(val_path)
        if include_test:
            test_ds = SpotsDataset.from_folder(test_path)
    else:
        tr_ds = SpotsDataset.from_folder(train_path, normalizer=None)
        val_ds = SpotsDataset.from_folder(val_path, normalizer=None)
        
        if include_test:
            test_ds = SpotsDataset.from_folder(test_path, normalizer=None)
    
    tr_imgs = tr_ds.images
    val_imgs = val_ds.images
    if include_test:
        test_imgs = test_ds.images

    tr_pts = tr_ds.centers
    val_pts = val_ds.centers
    if include_test:
        test_pts = test_ds.centers

    del tr_ds, val_ds

    if include_test:
        del test_ds
        return tr_imgs, tr_pts, val_imgs, val_pts, test_imgs, test_pts

    return tr_imgs, tr_pts, val_imgs, val_pts

def subpixel_offset(
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
    subpix = np.clip(subpix, -1, 1)
    n, _ = pts.shape
    _weight = np.zeros((n, 1), np.float32)
    _add = np.zeros((n, 2), np.float32)
    for i, j in product(range(-radius, radius + 1), repeat=2):
        dp = np.array([[i, j]])
        p = pts + dp
        # filter points outside of the image (boundary)
        p, mask = filter_shape(p, prob.shape, return_mask=True)
        _p = tuple(p.astype(int).T)

        _w = np.zeros((n, 1), np.float32)
        _w[mask] = prob[_p][:, None]

        _correct = np.zeros((n, 2), np.float32)
        _correct[mask] = subpix[_p] + dp

        _weight += _w
        _add += _w * _correct

    _add /= _weight
    return _add