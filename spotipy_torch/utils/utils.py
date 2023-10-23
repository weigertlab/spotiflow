import logging
import numpy as np
import os
import pandas as pd
import torch
import scipy.ndimage as ndi
import warnings
import wandb

from csbdeep.utils import normalize_mi_ma
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def read_coords_csv(fname: str):
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


def filter_shape(points, shape, idxr_array=None):
    """returns all values in "points" that are inside the shape as given by the indexer array
    if the indexer array is None, then the array to be filtered itself is used
    """
    if idxr_array is None:
        idxr_array = points.copy()
    assert idxr_array.ndim == 2 and idxr_array.shape[1] == 2
    idx = np.all(np.logical_and(idxr_array >= 0, idxr_array < np.array(shape)), axis=1)
    return points[idx]


def multiscale_decimate(y, decimate=(4, 4), sigma=1):
    if decimate == (1, 1):
        return y
    assert y.ndim == len(decimate)
    from skimage.measure import block_reduce

    y = block_reduce(y, decimate, np.max)
    y = 2 * np.pi * sigma**2 * ndi.gaussian_filter(y, sigma)
    y = np.clip(y, 0, 1)
    return y


def center_pad(x, shape, mode="reflect"):
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


def center_crop(x, shape):
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


def str2bool(v):
    if v.lower() in ("yes", "true", "True", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "False", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def str2scalar(dtype):
    def _f(v):
        if v.lower() == "none":
            return None
        else:
            return dtype(v)

    return _f


def normalize(
    x: np.ndarray, pmin=1, pmax=99.8, subsample: int = 1, clip=False, ignore_val=None
):
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
):
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


def initialize_wandb(options, train_dataset, val_dataset, silent=True):
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


def generate_datasets(
    X: Union[np.ndarray, Sequence],
    Y: Sequence,
    valX: Optional[Union[np.ndarray, Sequence]] = None,
    valY: Optional[Sequence] = None,
    val_frac: float = 0.15,
    seed: int = 42,
    **kwargs,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:

    """Generate the training and validation datasets with default settings.

    Args:
        X (Union[np.ndarray, Sequence]): training images
        Y (Sequence): training points
        valX (Union[np.ndarray, Sequence]): validation images. Optional
        valY (Sequence): validation labels. Optional
        val_frac (float, optional): fraction of the training set to use for validation. Unused if validation data is explicitly given. Defaults to .15.
        seed (int, optional): random seed for data splitting. Unused if validation data is explicitly given. Defaults to 42.
        kwargs: additional arguments to pass to the SpotsDataset class
    """

    from ..data import SpotsDataset

    if isinstance(X, np.ndarray):
        X = [X[i] for i in range(X.shape[0])]

    if valX is not None and isinstance(valX, np.ndarray):
        valX = [valX[i] for i in range(valX.shape[0])]
    elif valX is None:
        log.info(
            f"No validation data given, will use a subset of training data as validation ({val_frac:.2f})."
        )
        rng = np.random.default_rng(seed)
        val_idx = sorted(
            rng.choice(len(X), int(len(X) * val_frac), replace=False, shuffle=False)
        )
        valX = [X[i] for i in val_idx]
        valY = [Y[i] for i in val_idx]
        X = [X[i] for i in range(len(X)) if i not in val_idx]
        Y = [Y[i] for i in range(len(Y)) if i not in val_idx]

    train_ds = SpotsDataset(X, Y, **kwargs)

    # Avoid augmenting validation data
    kwargs_val = kwargs.copy()
    kwargs_val["augmenter"] = None
    val_ds = SpotsDataset(valX, valY, **kwargs)
    return train_ds, val_ds
