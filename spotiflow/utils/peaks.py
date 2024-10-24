from numbers import Number
from typing import Literal, Tuple, Union

import numpy as np
import numpy as np
from skimage.feature import corner_peaks, corner_subpix
from skimage.feature.peak import (
    _get_excluded_border_width,
    _get_threshold,
    _exclude_border,
)
import os
from .utils import filter_shape

from ..lib.filters import c_maximum_filter_2d_float
from ..lib.filters3d import c_maximum_filter_3d_float
from ..lib.point_nms import c_point_nms_2d
from ..lib.point_nms3d import c_point_nms_3d
from ..lib.spotflow2d import c_spotflow2d, c_gaussian2d
from ..lib.spotflow3d import c_spotflow3d, c_gaussian3d


def get_num_threads():
    # set OMP_NUM_THREADS to 1/2 of the number of CPUs by default
    n_cpu = os.cpu_count()
    n_threads = int(os.environ.get("OMP_NUM_THREADS", n_cpu))
    n_threads = max(1, min(n_threads, n_cpu // 2))
    return n_threads


def nms_points_2d(
    points: np.ndarray, scores: np.ndarray = None, min_distance: int = 2
) -> np.ndarray:
    """Non-maximum suppression for 2D points, choosing the highest scoring points while
    ensuring that no two points are closer than min_distance.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N,2) containing the points to be filtered.
    scores : np.ndarray
        Array of shape (N,) containing scores for each point
        If None, all points have the same score
    min_distance : int, optional
        Minimum distance between points, by default 2

    Returns
    -------
    np.ndarray
        Array of shape (N,) containing the indices of the points that survived the filtering.
    """

    points = np.asarray(points)
    if not points.ndim == 2 and points.shape[1] == 2:
        raise ValueError("points must be a array of shape (N,2)")
    if scores is None:
        scores = np.ones(len(points))
    else:
        scores = np.asarray(scores)
    if not scores.ndim == 1:
        raise ValueError("scores must be a array of shape (N,)")

    idx = np.argsort(scores, kind="stable")[::-1]
    points = points[idx]
    scores = scores[idx]
    
    points = np.ascontiguousarray(points, dtype=np.float32)
    inds = c_point_nms_2d(points, np.float32(min_distance))
    inds = idx[inds]
    return inds

def nms_points_3d(
    points: np.ndarray, scores: np.ndarray = None, min_distance: int = 2
) -> np.ndarray:
    """Non-maximum suppression for 2D points, choosing the highest scoring points while
    ensuring that no two points are closer than min_distance.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N,2) containing the points to be filtered.
    scores : np.ndarray
        Array of shape (N,) containing scores for each point
        If None, all points have the same score
    min_distance : int, optional
        Minimum distance between points, by default 2

    Returns
    -------
    np.ndarray
        Array of shape (N,) containing the indices of the points that survived the filtering.
    """

    points = np.asarray(points)
    if not points.ndim == 2 and points.shape[1] == 3:
        raise ValueError("points must be a array of shape (N,3)")
    if scores is None:
        scores = np.ones(len(points))
    else:
        scores = np.asarray(scores)
    if not scores.ndim == 1:
        raise ValueError("scores must be a array of shape (N,)")

    idx = np.argsort(scores, kind="stable")[::-1]
    points = points[idx]
    scores = scores[idx]

    points = np.ascontiguousarray(points, dtype=np.float32)
    inds = c_point_nms_3d(points, np.float32(min_distance))
    inds = idx[inds]
    return inds

def maximum_filter_2d(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:

    if not image.ndim == 2:
        raise ValueError("Image must be 2D")
    if not kernel_size > 0 and kernel_size % 2 == 1:
        raise ValueError("kernel_size must be positive and odd")

    image = np.ascontiguousarray(image, dtype=np.float32)
    n_threads = get_num_threads()
    return c_maximum_filter_2d_float(
        image, np.int32(kernel_size // 2), np.int32(n_threads)
    )

def maximum_filter_3d(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if not image.ndim == 3:
        raise ValueError("Image must be 3D")
    if not kernel_size > 0 and kernel_size % 2 == 1:
        raise ValueError("kernel_size must be positive and odd")

    image = np.ascontiguousarray(image, dtype=np.float32)
    n_threads = get_num_threads()
    return c_maximum_filter_3d_float(
        image, np.int32(kernel_size // 2), np.int32(n_threads)
    )


def points_to_prob(points, shape, sigma: Union[np.ndarray, float]=1.5, val:Union[np.ndarray, float]=1., mode:str ="max", grid: Union[int, Tuple[int,int,int]]=None) -> np.ndarray:
    """Wrapper function for different cpp calls. Points should be in (y,x) or (z,y,x) order"""

    ndim=len(shape) 
    
    points = np.asarray(points)
    
    if not points.shape[1] == ndim:
        raise ValueError("Wrong dimension of points!")

    if grid is not None and any(s != 1 for s in grid) and ndim == 2:
        raise NotImplementedError("grid not yet implemented for 2d")

    if grid is None:
        grid = (1, )*ndim
    elif isinstance(grid, int):
        grid = (grid, )*ndim

    if len(grid) != ndim:
        raise ValueError("grid must have the same dimension as shape")
    
    
    
    if ndim == 2:
        return points_to_prob2d(points, shape=shape, sigma=sigma, val=val, mode=mode)
    elif ndim == 3:
        return points_to_prob3d(points, shape=shape, sigma=sigma, grid=grid, mode=mode)
    else:
        raise ValueError("Wrong dimension of points!")

def points_to_prob2d(points, shape, 
                     sigma: Union[np.ndarray, float]=1.5,
                     val: Union[np.ndarray, float]=1., 
                     mode:Literal["max","sum"]="max") -> np.ndarray:
    """ 
    Create a 2D probability map from a set of points
    
    Parameters
    ----------
    points : np.ndarray
        Array of shape (N,2) containing the points to be filtered.
    shape : tuple
        shape of the output array
    sigma : float or list/array of floats 
        sigma of the gaussians, by default 1.5
    mode : str, optional
        mode of the filter, by default "max"
    val : float or list/array of floats 
        Value or array of shape (N,) containing the value at the center of each point, by default 1.
    """

    x = np.zeros(shape, np.float32)
    assert points.ndim == 2 and points.shape[1] == 2
    points = filter_shape(points, shape)

    if isinstance(sigma, Number):
        sigma = np.ones(len(points), np.float32) * sigma
    else: 
        sigma = np.asarray(sigma, np.float32)
    
    if isinstance(val, Number):
        val = np.ones(len(points), np.float32) * val
    else: 
        val = np.asarray(val, np.float32)
            
    if not len(points) == len(val) or not len(points) == len(sigma):
        raise ValueError("points, sigmas, and probs must have the same length")
            

    if len(points) == 0:
        return x

    if mode == "max":
        x = c_gaussian2d(
            points.astype(np.float32, copy=False),
            val.astype(np.float32, copy=False),
            sigma.astype(np.float32, copy=False),
            np.int32(shape[0]),
            np.int32(shape[1]),
        )
    elif mode == "sum":
        x = np.zeros(shape, np.float32)
        Y, X = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        for p, s, v in zip(points, sigma, val):
            x += v * np.exp(-((Y - p[0]) ** 2 + (X - p[1]) ** 2) / (2 * s ** 2))        
    else:
        raise ValueError(mode)

    return x

def points_to_prob3d(points, shape, 
                     sigma: Union[np.ndarray, float]=1.5,
                     val: Union[np.ndarray, float]=1., 
                     mode:Literal["max","sum"]="max",
                     grid: Union[int, Tuple[int,int,int]]=None):
    """points are in (z,y,x) order"""

    ndim=len(shape)
    
    assert len(grid) == ndim and all(isinstance(i, int) for i in shape) and all(isinstance(i, int) for i in grid), "shape and grid must be a 3-integer tuple"

    # TODO: needed?
    assert all(s%g == 0 for s, g in zip(shape, grid)), "shape must be divisible by grid"
    x = np.zeros(tuple(s//g for s, g in zip(shape, grid)), np.float32)
    assert points.ndim == 2 and points.shape[1] == ndim
    points = filter_shape(points, shape)

    if len(points) == 0:
        return x
    
    if isinstance(sigma, Number):
        sigma = np.ones(len(points), np.float32) * sigma
    else: 
        sigma = np.asarray(sigma, np.float32)
    
    if isinstance(val, Number):
        val = np.ones(len(points), np.float32) * val
    else: 
        val = np.asarray(val, np.float32)
    

    if mode == "max":
        x = c_gaussian3d(
            points.astype(np.float32, copy=False),
            val.astype(np.float32, copy=False),
            sigma.astype(np.float32, copy=False),            
            np.int32(shape[0]),
            np.int32(shape[1]),
            np.int32(shape[2]),
            np.int32(grid[0]),
            np.int32(grid[1]),
            np.int32(grid[2]),
        )
    elif mode == "sum":
        x = np.zeros(shape, np.float32)
        Xs = np.stack(np.meshgrid(*(np.arange(s) for s in shape), indexing="ij"))
        for p, s, v in zip(points, sigma, val):
            x += v * np.exp(- np.sum((Xs - p[:,None,None,None]) ** 2,axis=0) / (2 * s ** 2))        
        
    else:
        raise ValueError(mode)

    return x

def points_to_flow(points: np.ndarray, shape: tuple, sigma: float = 1.5, grid: Union[int, Tuple[int,int,int]]=None):
    """
    for each grid point in shape compute the vector d in R^N to the closest point
    and return its flow embedding onto S^(N+1)
    """
    ndim=len(shape)
    assert points.shape[-1] == ndim 
    if grid is None:
        grid = (1,)*ndim
    elif isinstance(grid, int):
        grid = (grid,)*ndim
    
    if ndim == 2:
        return points_to_flow2d(points, shape, sigma)
    elif ndim == 3:
        return points_to_flow3d(points, shape=shape, sigma=sigma, grid=grid)
    else:
        raise ValueError("Dimensionality of the input points should be 2 or 3!")


def points_to_flow2d(points: np.ndarray, shape: tuple, sigma: float = 1.5):
    """
    for each grid point in shape compute the vector d=(y,x) to the closest
    point and return its flow embedding (z',y',x') onto S^3

    z' = -(r^2 - sigma^2) / (r^2 + sigma^2);
    y' = 2 * sigma * y / (r^2 + sigma^2);
    x' = 2 * sigma * x / (r^2 + sigma^2);



    with r = sqrt(x^2+y^2)
    """
    if len(points) == 0:
        flow = np.zeros(shape[:2] + (3,), np.float32)
        flow[..., 0] = -1
        return flow
    else:
        return c_spotflow2d(
            points.astype(np.float32, copy=False),
            np.int32(shape[0]),
            np.int32(shape[1]),
            np.float32(sigma),
        )

def points_to_flow3d(points: np.ndarray, shape: tuple, sigma: float = 1.5, grid: Union[int, Tuple[int,int,int]] = (1,1,1)):
    """
    for each grid point in shape compute the vector d=(z,y,x) to the closest
    point and return its flow embedding (w',z',y',x') onto S^4

    w' = -(r^2 - sigma^2) / (r^2 + sigma^2);
    z' = 2 * sigma * z / (r^2 + sigma^2);
    y' = 2 * sigma * y / (r^2 + sigma^2);
    x' = 2 * sigma * x / (r^2 + sigma^2);



    with r = sqrt(x^2+y^2+z^2)
    """
    if isinstance(grid, int):
        grid = (grid, grid, grid)
    
    assert len(shape) == 3 and len(grid) == 3 and all(isinstance(i, int) for i in shape) and all(isinstance(i, int) for i in grid), "shape and grid must be a 3-integer tuple"
    assert all(s%g == 0 for s, g in zip(shape, grid)), "shape must be divisible by grid"
    if len(points) == 0:
        flow = np.zeros(tuple(s//g for s, g in zip(shape[:3], grid)) + (4,), np.float32)
        flow[..., 0] = -1
        return flow
    else:
        return c_spotflow3d(
            points.astype(np.float32, copy=False),
            np.int32(shape[0]),
            np.int32(shape[1]),
            np.int32(shape[2]),
            np.int32(grid[0]),
            np.int32(grid[1]),
            np.int32(grid[2]),
            np.float32(sigma),
        )


def flow_to_vector(flow: np.ndarray, sigma: float, eps: float=1e-20):
    ndim=flow.ndim-1
    if ndim == 2:
        return flow_to_vector_2d(flow, sigma=sigma, eps=eps)
    elif ndim == 3:
        return flow_to_vector_3d(flow, sigma=sigma, eps=eps)
    else:
        raise ValueError(f"Dimensionality of the stereographic flow should be 3 or 4!")

def flow_to_vector_2d(flow: np.ndarray, sigma: float, eps: float = 1e-20):
    """from the 3d flow (z',y',x') compute back the 2d vector field (y,x) it corresponds to

    y = sigma*y'/(1+z')
    x = sigma*x'/(1+z')

    """
    z, y, x = flow.transpose(2, 0, 1)
    s = sigma / (1 + z + eps)
    return np.stack((y * s, x * s), axis=-1)


def flow_to_vector_3d(flow: np.ndarray, sigma: float, eps: float = 1e-20):
    """from the 4d flow (w',z',y',x') compute back the 3d vector field (z,y,x) it corresponds to

    Args:
        flow (np.ndarray): stereographic flow of shape (w',z',y',x')
        sigma (float): scale length of the flow
        eps (float, optional): epsilon for numerical stability. Defaults to 1e-20.
    """
    w, z, y, x = flow.transpose(3, 0, 1, 2)
    s = sigma / (1 + w + eps)
    offsets = np.stack((z * s, y * s, x * s), axis=-1)
    return offsets

def prob_to_points(
    prob,
    prob_thresh=0.5,
    min_distance=2,
    subpix: bool = False,
    mode: str = "skimage",
    exclude_border: bool = True,
):
    assert prob.ndim in (2, 3), "Wrong dimension of prob"
    if mode == "skimage":
        corners = corner_peaks(
            prob,
            min_distance=min_distance,
            threshold_abs=prob_thresh,
            threshold_rel=0,
            exclude_border=exclude_border,
        )
        if subpix:
            print("using subpix")
            corners_sub = corner_subpix(prob, corners, window_size=3)
            ind = ~np.isnan(corners_sub[:, 0])
            corners[ind] = corners_sub[ind].round().astype(int)
    elif mode == "fast":
        corners = local_peaks(
            prob,
            min_distance=min_distance,
            threshold_abs=prob_thresh,
            exclude_border=exclude_border,
        )
    else:
        raise NotImplementedError(f'unknown mode {mode} (supported: "skimage", "fast")')
    return corners


def local_peaks(
    image: np.ndarray,
    min_distance=1,
    exclude_border=True,
    threshold_abs=None,
    threshold_rel=None,
    use_score:bool=False
):
    if not image.ndim in [2, 3]:
        raise ValueError("Image must be 2D")
    max_filter_fun = maximum_filter_2d if image.ndim == 2 else maximum_filter_3d
    nms_fun = nms_points_2d if image.ndim == 2 else nms_points_3d

    # make compatible with scikit-image
    # https://github.com/scikit-image/scikit-image/blob/a4e533ea2a1947f13b88219e5f2c5931ab092413/skimage/feature/peak.py#L120
    border_width = _get_excluded_border_width(image, min_distance, exclude_border)
    threshold = _get_threshold(image, threshold_abs, threshold_rel)

    image = image.astype(np.float32)

    if min_distance <= 0:
        mask = image > threshold
    else:
        mask = max_filter_fun(image, 2 * min_distance + 1) == image

        # no peak for a trivial image
        image_is_trivial = np.all(mask)
        if image_is_trivial:
            mask[:] = False
        mask &= image > threshold

    mask = _exclude_border(mask, border_width)

    coord = np.nonzero(mask)
    coord = np.stack(coord, axis=1)

    if use_score:
        scores = image[mask] if mask.sum() > 0 else None
    else:
        scores = None
        
    idx = nms_fun(coord, scores=scores, min_distance=min_distance)
    coord = coord[idx].copy()
    return coord



def points_from_heatmap_flow(heatmap:np.ndarray, flow:np.ndarray, sigma:float, grid:tuple[int]=None, local_peak_kwargs=None):
    """ Returns the points from the heatmap and flow field""" 
    ndim = heatmap.ndim

    if not flow.ndim == ndim+1:
        raise ValueError("Flow and heatmap must have the same dimension")
    if not flow.shape[-1] == ndim+1:
        raise ValueError(f"Last dimension of flow must be {ndim+1}")
    
    if grid is None:
        grid = (1,)*ndim

    grid = np.asarray(grid)

    if local_peak_kwargs is None:
        local_peak_kwargs = {}

    points_new = local_peaks(heatmap, **local_peak_kwargs)
    flow_reversed = flow_to_vector(flow, sigma=sigma)    
    offsets = flow_reversed[tuple(points_new.T.astype(int))]
    points_new = grid[None]*(points_new + offsets)
    return points_new