import argparse
import logging
import os
import sys
from itertools import product
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import dask
import dask.array as da
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch
import wandb
from csbdeep.utils import normalize_mi_ma
from skimage.io import imread
from torch.utils.data import Dataset

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

def imread_wrapped(fname, channels=None, zarr_component=None):
    """Note that multi-channel images have to have the channels in the last dimension."""
    try:
        if fname.suffix == ".zarr":
            img = da.from_zarr(fname, component=zarr_component)
        else:
            img = imread(fname)
        if channels is not None:
            if any(c >= img.shape[-1] for c in channels):
                raise ValueError(
                    f"Some of the requested channels {channels} are not present in the image {fname} with shape {img.shape}"
                )
            if len(channels) == 1:
                img = img[..., channels[0]]
            else:
                img = img[..., channels]
        return img
    except Exception as e:
        log.error(f"Could not read image {fname}. Execution will halt.")
        raise e


def str2bool(value):
    value = value.lower()
    if value in ("yes", "true", "t", "y", "1"):
        return True
    elif value in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"{value}: Boolean value expected!")


def infer_n_tiles(
    shape: Tuple[int],
    max_tile_size: Optional[Tuple[int]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int]:
    """
    infers number of tiles by dimension
    """
    should_auto_infer = max_tile_size is None
    if should_auto_infer:
        fallback_tile_size = ((2048, 2048), (128, 256, 256))
        max_tile_size = (
            fallback_tile_size[0] if len(shape) == 2 else fallback_tile_size[1]
        )
        if device is not None:
            if device.type == "cuda":
                gpu_mem_gb = torch.cuda.mem_get_info(device=device)[1] / 1e9
                mem_to_size = {
                    1: ((512, 512), (64, 64, 64)),
                    2: ((512, 1024), (64, 128, 128)),
                    4: ((1024, 1024), (128, 128, 128)),
                    8: ((1024, 2048), (64, 256, 256)),
                    16: ((2048, 2048), (128, 256, 256)),
                    32: ((2048, 4096), (256, 256, 256)),
                    64: ((4096, 4096), (128, 512, 512)),
                }
                # Get closest power of 2 that is less than or equal to the available GPU memory
                floored_gpu_mem_gb = 2 ** int(np.log2(gpu_mem_gb))
                max_tile_size_both = mem_to_size.get(
                    floored_gpu_mem_gb, fallback_tile_size
                )
                max_tile_size = (
                    max_tile_size_both[0] if len(shape) == 2 else max_tile_size_both[1]
                )
    if not len(shape) == len(max_tile_size):
        raise ValueError(
            f"shape {shape} and max_tile_size {max_tile_size} should have the same length"
        )
    return tuple(int(np.ceil(s / m)) for s, m in zip(shape, max_tile_size))


# TODO: add_class_column is set to False for now not to break downstream code, but it shouldn't even be a parameter
def read_coords_csv(fname: str, add_class_column: bool = False) -> np.ndarray:
    """Parses a csv file and returns correctly ordered points array

    Args:
        fname (str): Path to the csv file
        add_class_column (bool, optional): Whether to add a class column to the points array. Defaults to False.
    Returns:
        np.ndarray: A 2D array of spot coordinates. If `add_class_column` is True, then the array will have shape (N, 3) where N is the number of points. Otherwise, the array will have shape (N, 2)
    """
    try:
        df = pd.read_csv(fname)
    except pd.errors.EmptyDataError:
        return np.zeros((0, 2 if not add_class_column else 3), dtype=np.float32)

    df = df.rename(columns=str.lower)
    df = df.rename(columns=str.strip)
    cols = set(df.columns)

    col_candidates = (("axis-0", "axis-1"), ("y", "x"), ("Y", "X"))
    points = None
    for possible_columns in col_candidates:
        if cols.issuperset(set(possible_columns)):
            points = df[list(possible_columns)].to_numpy()
            break

    if add_class_column:
        class_col_candidates = ("class", "label", "category", "channel")

        class_labels = np.zeros((points.shape[0], 1), dtype=np.float32)
        for possible_class_column in class_col_candidates:
            if possible_class_column in cols:
                class_labels = (
                    df[possible_class_column].to_numpy().reshape(-1, 1).astype(np.uint8)
                )
                break
        points = np.concatenate((points, class_labels), axis=1)

    if points is None:
        raise ValueError(f"could not get points from csv file {fname}")

    return points


def read_coords_csv3d(fname: str) -> np.ndarray:
    """Parses a csv file and returns correctly ordered points array

    Args:
        fname (str): Path to the csv file
    Returns:
        np.ndarray: A 3D array of spot coordinates
    """
    try:
        df = pd.read_csv(fname)
    except pd.errors.EmptyDataError:
        return np.zeros((0, 3), dtype=np.float32)

    df = df.rename(columns=str.lower)
    df = df.rename(columns=str.strip)
    cols = set(df.columns)

    col_candidates = (("axis-0", "axis-1", "axis-2"), ("z", "y", "x"), ("Z", "Y", "X"))
    points = None
    for possible_columns in col_candidates:
        if cols.issuperset(set(possible_columns)):
            points = df[list(possible_columns)].to_numpy()
            break

    if points is None:
        raise ValueError(f"could not get points from csv file {fname}")

    return points


def filter_shape(
    points: np.ndarray,
    shape: Tuple[int, int],
    idxr_array: Optional[np.ndarray] = None,
    return_mask: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Returns all values in "points" that are inside the shape as given by the indexer array
    if the indexer array is None, then the array to be filtered itself is used

    Args:
        points (np.ndarray): 2D array of points to be filtered
        shape (Tuple[int, int]): Shape of the image. Points outside this shape will be filtered out.
        idxr_array (Optional[np.ndarray], optional): Array to be used for filtering. If None, uses the input array itself. Defaults to None.
        return_mask (bool, optional): Whether to return the boolean mask. Defaults to False.
    """
    if idxr_array is None:
        idxr_array = points.copy()
    assert idxr_array.ndim == 2 and idxr_array.shape[1] in (2, 3)
    idx = np.all(np.logical_and(idxr_array >= 0, idxr_array < np.array(shape)), axis=1)
    if return_mask:
        return points[idx], idx
    return points[idx]


def multiscale_decimate(
    y: np.ndarray,
    decimate: Tuple[int, int] = (2, 2),
    sigma: float = 1.0,
    is_3d: bool = False,
) -> np.ndarray:
    """Decimate an image by a factor of `decimate` and apply a Gaussian filter with standard deviation `sigma`

    Args:
        y (np.ndarray): Image to be decimated
        decimate (Tuple[int, int], optional): downsampling factor. Defaults to (2, 2).
        sigma (float, optional): standard deviation of the Gaussian filter. Defaults to 1.

    Returns:
        np.ndarray: Decimated image
    """
    from skimage.measure import block_reduce

    if is_3d:
        if len(decimate) == 2 and y.ndim == 3:  # 3D Image
            decimate = (decimate[0], *decimate)
    else:
        if len(decimate) == 2 and y.ndim == 3:  # Multichannel image
            decimate = (1, *decimate)
    assert y.ndim == len(decimate), (
        f"decimate {decimate} and y.ndim {y.ndim} do not match"
    )
    if decimate == (1, 1) or decimate == (1, 1, 1):
        return y

    y = block_reduce(y, decimate, np.max)
    y = (
        2
        * np.pi
        * sigma**2
        * ndi.gaussian_filter(y, sigma, axes=(-2, -1) if not is_3d else (-3, -2, -1))
    )
    y = np.clip(y, 0, 1)
    return y


def center_pad(
    x: Union[np.ndarray, da.Array],
    shape: Tuple[int, int],
    mode: str = "reflect",
    allow_larger: bool = False,
) -> Tuple[Union[np.ndarray, da.Array], Sequence[Tuple[int, int]]]:
    """Pads x to shape. This function is the inverse of center_crop.
       This function accepts both NumPy arrays and Dask arrays. For the latter, the padding is done lazily.

    Args:
        x (Union[np.ndarray, da.Array]): Image to be padded
        shape (Tuple[int, int]): Shape of the padded image
        mode (str, optional): Padding mode. Defaults to "reflect".

    Returns:
        Tuple[Union[np.ndarray, da.Array], Sequence[Tuple[int, int]]]: A tuple of the padded image and the padding sequence
    """
    if x.shape == shape or (
        allow_larger and all([s1 > s2 for s1, s2 in zip(x.shape, shape)])
    ):
        return x, tuple((0, 0) for _ in x.shape)
    if not allow_larger and not all([s1 <= s2 for s1, s2 in zip(x.shape, shape)]):
        raise ValueError(f"shape of x {x.shape} is larger than final shape {shape}")
    diff = np.array(shape) - np.array(x.shape)
    pads = tuple(
        (int(np.ceil(d / 2)), d - int(np.ceil(d / 2))) if d > 0 else (0, 0)
        for d in diff
    )
    _pad_func = da.pad if isinstance(x, da.Array) else np.pad
    return _pad_func(x, pads, mode=mode), pads


def center_crop(
    x: Union[np.ndarray, da.Array], shape: Tuple[int, int]
) -> Union[np.ndarray, da.Array]:
    """Crops x to given shape. This function is the inverse of center_pad

    y = center_pad(x,shape)
    z = center_crop(y,x.shape)
    np.allclose(x,z)

    Args:
        x (np.ndarray): Image to be cropped
        shape (Tuple[int, int]): Shape of the cropped image

    Returns:
        np.ndarray: Cropped image
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


def normalize(
    x: np.ndarray,
    pmin: float = 1.0,
    pmax: float = 99.8,
    subsample: int = 1,
    clip: bool = False,
    ignore_val: Optional[Union[int, float]] = None,
) -> np.ndarray:
    """
    Normalizes (percentile-based) a 2d image with the additional option to ignore a value. The normalization is done as follows:

    x = (x - I_{p_{min}}) / (I_{p_{max}} - I_{p_{min}})

    where I_{p_{min}} and I_{p_{max}} are the pmin and pmax percentiles of the image intensity, respectively.

    Args:
        x (np.ndarray): Image to be normalized
        pmin (float, optional): Minimum percentile. Defaults to 1..
        pmax (float, optional): Maximum percentile. Defaults to 99.8.
        subsample (int, optional): Subsampling factor for percentile calculation. Defaults to 1.
        clip (bool, optional): Whether to clip the normalized image. Defaults to False.
        ignore_val (Optional[Union[int, float]], optional): Value to be ignored. Defaults to None.

    Returns:
        np.ndarray: Normalized image
    """
    if isinstance(x, da.Array):
        raise TypeError("Please use the `normalize_dask` function for Dask arrays!")

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

def dask_normalize_mi_ma(
    x: da.Array,
    mi: float,
    ma: float,
    clip: bool=False,
    eps: float=1e-20,
    dtype: np.dtype=np.float32
) -> da.Array:
    """
    Normalize a Dask array using min-max normalization with the given values.
    Adapted from csbdeep.utils.normalize_mi_ma to work with Dask array (removed numexpr).
    """
    if not isinstance(x, da.Array):
        raise TypeError("Input x should be a Dask array. Please use the `normalize`/`normalize_mi_ma` function for NumPy arrays.")
    if dtype is not None:
        x = x.astype(dtype)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    x = (x - mi) / (ma - mi + eps)

    if clip:
        x = x.clip(0, 1)
    return x



def normalize_dask(
    x: da.Array,
    pmin: float = 1.0,
    pmax: float = 99.8,
    eps: float = 1e-20,
    max_samples: int = 1e5,
    clip: bool = False,
    dtype: Optional[np.dtype] = np.float32,
) -> da.Array:
    """
    Lazily normalizes (percentile-based) an n-dimensional Dask array with the additional option to ignore a value. The normalization is done as follows:

    x = (x - I_{p_{min}}) / (I_{p_{max}} - I_{p_{min}})

    where I_{p_{min}} and I_{p_{max}} are the pmin and pmax percentiles of the image intensity, respectively.

    Args:
        x (da.Array): array to be normalized
        pmin (float, optional): Minimum percentile. Defaults to 1..
        pmax (float, optional): Maximum percentile. Defaults to 99.8.
        eps (float, optional): Epsilon value to avoid division by zero. Defaults to 1e-20.
        max_samples (int, optional): Maximum number of samples to use for percentile calculation. Defaults to 1e5.

    Returns:
        da.Array: lazily normalized image
    """

    if isinstance(x, np.ndarray):
        raise TypeError("Please use the `normalize` function for NumPy arrays!")

    n_skip = int(max(1, x.size // max_samples))
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        mi, ma = da.percentile(
            x.ravel()[::n_skip], (pmin, pmax), internal_method="tdigest"
        ).compute()
    return dask_normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def initialize_wandb(
    options: dict, train_dataset: Dataset, val_dataset: Dataset, silent: bool = True
) -> None:
    """Helper function which initializes wandb for logging. If `options` contains the key `skip_logging`, then wandb will not be initialized.

    Args:
        options (dict): Dictionary containing the options for wandb
        train_dataset (Dataset): Training dataset
        val_dataset (Dataset): Validation dataset
        silent (bool, optional): Whether to suppress wandb output to stdout/stderr. Defaults to True.
    """

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


def write_coords_csv(pts: np.ndarray, fname: Union[Path, str]) -> None:
    """Writes points in a NumPy array to a CSV file

    Args:
        pts (np.ndarray): 2D or 3D array of points
        fname (Union[Path, str]): Path to the CSV file
    """
    columns = ("y", "x")
    if pts.shape[1] == 3:
        # 3D, add leading "z"
        columns = ("z",) + columns

    df = pd.DataFrame(pts, columns=columns)
    df.to_csv(fname, index=False)
    return


def remove_device_id_from_device_str(device_str: str) -> str:
    """Helper function to remove the device id from the device string.
    For example, "cuda:0" will be converted to "cuda"

    Args:
        device_str (str): Device string

    Returns:
        str: Device string without the device id
    """
    return device_str.split(":")[0].strip()


def get_data(
    path: Union[Path, str],
    normalize: bool = True,
    include_test: bool = False,
    subfolder: Optional[tuple[str]] = None, # ("train", "val", "test"),
    is_3d: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get data from a given root path and subfolders.

    Args:
        path (Union[Path, str]): Path to the data.
        normalize (bool, optional): Whether to normalize the data. Defaults to True.
        include_test: (bool, optional): Whether to include the test set. Defaults to False.
        subfolder (tuple[str], optional): Subfolders to be used for non-standard datasets. Defaults to None.
    Returns:
        Tuple[np.ndarray]: A tuple of arrays with length 2*len(subfolder) corresponding to the images, centers per subfolder
    """
    from ..data import Spots3DDataset, SpotsDataset

    SpotsDatasetClass = SpotsDataset if not is_3d else Spots3DDataset

    path = Path(path)

    if subfolder is None and not include_test:
        subfolder = ("train", "val")
    elif subfolder is None and include_test:
        subfolder = ("train", "val", "test")

    if not path.exists():
        raise FileNotFoundError(f"Given data path {path} does not exist!")
    for sub in subfolder:
        if not (path / sub).exists():
            raise FileNotFoundError(
                f"Given data path {path} does not contain a '{sub}' folder!"
            )

    if not normalize and not "normalizer" in kwargs:
        kwargs["normalizer"] = None

    datasets = tuple(
        SpotsDatasetClass.from_folder(path / sub, add_class_label=False, **kwargs)
        for sub in subfolder
    )
    results = tuple(x for d in datasets for x in (d.images, d.centers))
    return results


def subpixel_offset_2d(
    pts: np.ndarray,
    subpix: np.ndarray,
    prob: np.ndarray,
    radius: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute offset vector for subpixel localization at given locations by aggregating within a radius the
    2D local vector field `subpix` around each point in `pts` weighted by the probability array `prob`

    Args:
        pts (np.ndarray): 2D array of points of shape (N, 2)
        subpix (np.ndarray): local vector field in Euclidean space. Should be a 3D array with shape (H, W, 2)
        prob (np.ndarray): 2D array of probabilities of shape (H, W)
        radius (int): Radius for aggregation
        eps (float, optional): Epsilon value to avoid division by zero. Defaults to 1e-8.

    Returns:
        np.ndarray: 2D array of offsets
    """
    assert (
        pts.ndim == 2
        and pts.shape[1] == 2
        and subpix.ndim == 3
        and subpix.shape[2] == 2
        and prob.ndim == 2
    )
    subpix_cp = subpix.copy()
    subpix_cp[np.linalg.norm(subpix_cp, axis=-1) > np.sqrt(2)] = 0
    n, _ = pts.shape
    _weight = np.zeros((n, 1), np.float32)
    _add = np.zeros((n, 2), np.float32)
    for i, j in product(range(-radius, radius + 1), repeat=2):
        dp = np.array([[i, j]])
        p = pts + dp
        # filter points outside of the image (boundary)
        p, mask = filter_shape(p, prob.shape, return_mask=True)
        _p = tuple(p.astype(int).T)

        _w = np.zeros((n, 1), np.float32) + eps
        _w[mask] = prob[_p][:, None]

        _correct = np.zeros((n, 2), np.float32)
        _correct[mask] = subpix_cp[_p] + dp

        _weight += _w
        _add += _w * _correct

    _add /= _weight
    return _add


def subpixel_offset_3d(
    pts: np.ndarray,
    subpix: np.ndarray,
    prob: np.ndarray,
    radius: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute offset vector for subpixel localization at given locations by aggregating within a radius the
    3D local vector field `subpix` around each point in `pts` weighted by the probability array `prob`

    Args:
        pts (np.ndarray): 2D array of points of shape (N, 3)
        subpix (np.ndarray): local vector field in Euclidean space. Should be a 4D array with shape (D, H, W, 2)
        prob (np.ndarray): heatmap of probabilities of shape (D, H, W)
        radius (int): Radius for aggregation
        eps (float, optional): Epsilon value to avoid division by zero. Defaults to 1e-8.

    Returns:
        np.ndarray: 3D array of offsets
    """
    assert (
        pts.ndim == 2
        and pts.shape[1] == 3
        and subpix.ndim == 4
        and subpix.shape[3] == 3
        and prob.ndim == 3
    )
    subpix_cp = subpix.copy()
    subpix_cp[np.linalg.norm(subpix, axis=-1) > np.sqrt(3)] = 0
    n, _ = pts.shape
    _weight = np.zeros((n, 1), np.float32)
    _add = np.zeros((n, 3), np.float32)
    for i, j, k in product(range(-radius, radius + 1), repeat=3):
        dp = np.array([[i, j, k]])
        p = pts + dp
        # filter points outside of the image (boundary)
        p, mask = filter_shape(p, prob.shape, return_mask=True)
        _p = tuple(p.astype(int).T)

        _w = np.zeros((n, 1), np.float32) + eps
        _w[mask] = prob[_p][:, None]

        _correct = np.zeros((n, 3), np.float32)
        _correct[mask] = subpix[_p] + dp

        _weight += _w
        _add += _w * _correct

    _add /= _weight
    return _add


def subpixel_offset(
    pts: np.ndarray,
    subpix: np.ndarray,
    prob: np.ndarray,
    radius: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute offset vector for subpixel localization at given locations by aggregating within a radius the
    Euclidean local vector field `subpix` around each point in `pts` weighted by the probability array `prob`

    Args:
        pts (np.ndarray): 2D or 3D array of points of shape (N, 3)
        subpix (np.ndarray): local vector field in Euclidean space. Should be a 3D array with shape (H, W, 2) or (H, W, 3)
        prob (np.ndarray): 2D or 3D array of probabilities of shape ((D), H, W)
        radius (int): Radius for aggregation

    Returns:
        np.ndarray: 2D or 3D array of offsets
    """
    if pts.shape[1] == 2:
        return subpixel_offset_2d(pts, subpix, prob, radius, eps)
    elif pts.shape[1] == 3:
        return subpixel_offset_3d(pts, subpix, prob, radius, eps)
    else:
        raise ValueError(f"Invalid shape {pts.shape} for points array.")


def read_npz_dataset(fname: Union[Path, str]) -> Tuple[np.ndarray, ...]:
    """Reads a spots dataset from a .npz file formatted as in deepBlink (Eichenberger et al. 2021))

    Args:
        fname (Union[Path, str]): Path to the .npz file

    Returns:
        Tuple[np.ndarray, ...]: A 6-length tuple corresponding to training images, training spots, validation images,
                                validation spots, test images and test spots. Images are NumPy arrays of shape (N_i, H, W),
                                while spots is an N_i-element list of 2D arrays of shape (N_p, 2)
    """
    if isinstance(fname, str):
        fname = Path(fname)
    assert fname.suffix == ".npz", f"Given file {fname} is not a .npz file!"

    expected_keys = ["x_train", "y_train", "x_valid", "y_valid", "x_test", "y_test"]
    data = np.load(fname, allow_pickle=True)
    assert set(expected_keys).issubset(data.files), (
        f"Given .npz file {fname} does not contain the expected keys {expected_keys}!"
    )
    ret_data = [None] * len(expected_keys)
    for i, key in enumerate(expected_keys):
        if key.startswith("x"):
            ret_data[i] = np.asarray(data[key])
        elif key.startswith("y"):
            ret_data[i] = list(data[key])
        else:
            raise ValueError(f"Unexpected key {key} in .npz file {fname}")
    return ret_data


def spline_interp_points_2d(
    img: np.ndarray, pts: np.ndarray, order: int = 1
) -> np.ndarray:
    """Return the spline-interpolated 2D image intensities at each (subpixel) location.


    Args:
        img (np.ndarray): image in YX or YXC format.
        pts (np.ndarray): spot locations to interpolate the intensities from. Array shape should be (N,2).
        order (int, optional): order of the spline interpolation. Defaults to 1 (linear).

    Returns:
        np.ndarray: array of shape (N,C) containing intensities for each spot
    """
    assert img.ndim in (2, 3), "Expected YX or YXC image for interpolating intensities."
    assert pts.shape[1] == 2, (
        "Point coordinates to be interpolated should be an (N,2) array"
    )

    if pts.shape[0] == 0:
        if img.ndim == 2:
            out_shape = (0,)
        else:
            out_shape = (0, img.shape[-1])
        return np.zeros(out_shape, dtype=img.dtype)

    y_coords = pts[:, 0]
    x_coords = pts[:, 1]
    if img.ndim == 3:
        intensities = np.stack(
            [
                ndi.map_coordinates(
                    img[..., c],
                    [y_coords, x_coords],
                    order=order,
                    mode="reflect",
                    prefilter=False,
                )
                for c in range(img.shape[2])
            ],
            axis=-1,
        )
    else:
        intensities = ndi.map_coordinates(
            img, [y_coords, x_coords], order=order, mode="reflect", prefilter=False
        )
    return intensities


def spline_interp_points_3d(
    img: np.ndarray, pts: np.ndarray, order: int = 1
) -> np.ndarray:
    """Return the spline-interpolated 3D image intensities at each (subpixel) location.


    Args:
        img (np.ndarray): image in ZYX or ZYXC format.
        pts (np.ndarray): spot locations to interpolate the intensities from. Array shape should be (N,3).
        order (int, optional): order of the spline interpolation. Defaults to 1 (linear).

    Returns:
        np.ndarray: array of shape (N,C) containing intensities for each spot
    """
    assert img.ndim in (
        3,
        4,
    ), "Expected ZYX or ZYXC image for interpolating intensities."
    assert pts.shape[1] == 3, (
        "Point coordinates to be interpolated should be an (N,3) array"
    )

    if pts.shape[0] == 0:
        if img.ndim == 3:
            out_shape = (0,)
        else:
            out_shape = (0, img.shape[-1])
        return np.zeros(out_shape, dtype=img.dtype)

    zs, ys, xs = pts[:, 0], pts[:, 1], pts[:, 2]

    if img.ndim == 4:
        intensities = np.stack(
            [
                ndi.map_coordinates(
                    img[..., c],
                    [zs, ys, xs],
                    order=order,
                    mode="reflect",
                    prefilter=False,
                )
                for c in range(img.shape[3])
            ],
            axis=-1,
        )
    else:
        intensities = ndi.map_coordinates(
            img, [zs, ys, xs], order=1, mode="reflect", prefilter=False
        )
    return intensities
