import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from scipy.optimize import curve_fit
from tqdm.auto import tqdm

FWHM_CONSTANT = 2 * np.sqrt(2 * np.log(2))

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)


def _gaussian_2d(yx, y0, x0, sigma, A, B):
    y, x = yx
    return A * np.exp(-((y - y0) ** 2 + (x - x0) ** 2) / (2 * sigma**2)) + B


def _estimate_sigma_single(center: np.ndarray, image: np.ndarray, window: int = 5):
    x_range = np.arange(-window, window + 1)
    y_range = np.arange(-window, window + 1)
    y, x = np.meshgrid(y_range, x_range, indexing="ij")

    # Crop around the spot
    region = image[
        center[0] - window : center[0] + window + 1,
        center[1] - window : center[1] + window + 1,
    ]

    try:
        mi, ma = np.percentile(region, (0.1, 99.99))

        region = (region - mi) / (ma - mi)

        initial_guess = (0, 0, 1.5, 1, 0)  # y0, x0, sigma, A, B

        lower_bounds = (-2, -2, 0.1, -1, -1)  # y0, x0, sigma, A, B
        upper_bounds = (2, 2, 10, 10, 1)  # y0, x0, sigma, A, B

        popt, _ = curve_fit(
            _gaussian_2d,
            (y.ravel(), x.ravel()),
            region.ravel(),
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
        )
    except Exception as _:
        log.warning("Gaussian fit failed. Returning NaN")

    return popt[2]


def fwhm(sigma):
    return FWHM_CONSTANT * sigma


def estimate_fwhm(
    img: np.ndarray, centers: np.ndarray, window: int = 5, max_workers: int = 1
) -> np.ndarray:
    """Estimates FWHM of all spots in an image (given by centers) of the image by fitting a 2D Gaussian
       centered at each spot.

    Args:
        img (np.ndarray): input image
        centers (np.ndarray): centers of the spots to estimate FWHM of
        window (int, optional): window size around the spot to consider during the fitting. Defaults to 5.
        max_workers (int, optional): number of workers for parallelization. Defaults to 1.

    Returns:
        np.ndarray: estimated FWHM of the spots
    """

    img = np.pad(img, window, mode="reflect")
    centers = np.asarray(centers).astype(int) + window
    if max_workers == 1:
        sigmas = tuple(
            _estimate_sigma_single(p, image=img, window=window)
            for p in tqdm(centers, desc="Estimating FWHM of spots")
        )
    else:
        _partial_estimate_single_sigma = partial(
            _estimate_sigma_single, image=img, window=window
        )
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            sigmas = tuple(
                tqdm(
                    executor.map(_partial_estimate_single_sigma, centers),
                    total=len(centers),
                    desc="Estimating FWHM of spots",
                )
            )
    return fwhm(np.array(sigmas))
