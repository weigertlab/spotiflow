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
from ..lib.point_nms import c_point_nms_2d
from ..lib.spotflow2d import c_spotflow2d, c_gaussian2d


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

    idx = np.argsort(scores, kind="stable")
    points = points[idx]
    scores = scores[idx]

    points = np.ascontiguousarray(points, dtype=np.float32)
    inds = c_point_nms_2d(points, np.float32(min_distance))
    return points[inds].copy()


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


def points_to_prob(points, shape, sigma=1.5, mode="max"):
    """points are in (y,x) order"""

    x = np.zeros(shape, np.float32)
    assert points.ndim == 2 and points.shape[1] == 2
    points = filter_shape(points, shape)

    if len(points) == 0:
        return x

    if mode == "max":
        x = c_gaussian2d(
            points.astype(np.float32, copy=False),
            np.int32(shape[0]),
            np.int32(shape[1]),
            np.float32(sigma),
        )
    else:
        raise ValueError(mode)

    return x


def points_to_flow(points: np.ndarray, shape: tuple, sigma: float = 1.5):
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


def flow_to_vector(flow: np.ndarray, sigma: float, eps: float = 1e-20):
    """from the 3d flow (z',y',x') compute back the 2d vector field (y,x) it corresponds to

    y = sigma*y'/(1+z')
    x = sigma*x'/(1+z')

    """
    z, y, x = flow.transpose(2, 0, 1)
    s = sigma / (1 + z + eps)
    return np.stack((y * s, x * s), axis=-1)


def prob_to_points(
    prob,
    prob_thresh=0.5,
    min_distance=2,
    subpix: bool = False,
    mode: str = "skimage",
    exclude_border: bool = True,
):
    assert prob.ndim == 2, "Wrong dimension of prob"
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
):
    if not image.ndim == 2 and not image.ndim == 2:
        raise ValueError("Image must be 2D")

    # make compatible with scikit-image
    # https://github.com/scikit-image/scikit-image/blob/a4e533ea2a1947f13b88219e5f2c5931ab092413/skimage/feature/peak.py#L120
    border_width = _get_excluded_border_width(image, min_distance, exclude_border)
    threshold = _get_threshold(image, threshold_abs, threshold_rel)

    image = image.astype(np.float32)

    if min_distance <= 0:
        mask = image > threshold
    else:
        mask = maximum_filter_2d(image, 2 * min_distance + 1) == image

        # no peak for a trivial image
        image_is_trivial = np.all(mask)
        if image_is_trivial:
            mask[:] = False
        mask &= image > threshold

    mask = _exclude_border(mask, border_width)

    coord = np.nonzero(mask)
    coord = np.stack(coord, axis=1)

    points = nms_points_2d(coord, min_distance=min_distance)
    return points
