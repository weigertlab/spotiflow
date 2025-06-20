import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Union

import numpy as np
from scipy.optimize import curve_fit

from tqdm.auto import tqdm
from dataclasses import dataclass, fields
from scipy.ndimage import map_coordinates

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

def _gaussian_3d(zyx, z0, y0, x0, sigmaz, sigmayx, A, B):
    z, y, x = zyx
    return A * np.exp(
        -(((z - z0) ** 2) / (2 * sigmaz**2) + ((y - y0) ** 2 + (x - x0) ** 2) / (2 * sigmayx**2))
    ) + B

@dataclass
class FitParams2D:
    fwhm: Union[float, np.ndarray]
    offset_y: Union[float, np.ndarray]
    offset_x: Union[float, np.ndarray]
    intens_A: Union[float, np.ndarray]
    intens_B: Union[float, np.ndarray]
    r_squared: Union[float, np.ndarray]
    
@dataclass
class FitParams3D:
    fwhm_z: Union[float, np.ndarray]
    fwhm_yx: Union[float, np.ndarray]
    offset_z: Union[float, np.ndarray]
    offset_y: Union[float, np.ndarray]
    offset_x: Union[float, np.ndarray]
    intens_A: Union[float, np.ndarray]
    intens_B: Union[float, np.ndarray]
    r_squared: Union[float, np.ndarray]


def signal_to_background(params: FitParams2D) -> np.ndarray:
    """Calculates the signal to background ratio of the spots. Given a Gaussian fit
         of the form A*exp(...) + B, the signal to background
         ratio is computed as A/B.

    Args:
        params (SpotParams): SpotParams object containing the parameters of the Gaussian fit.

    Returns:
        np.ndarray: signal to background ratio of the spots
    """
    nonzero_mask = params.intens_B > 1e-12
    snb = np.full(params.intens_A.shape, np.inf)
    snb[nonzero_mask] = params.intens_A[nonzero_mask] / params.intens_B[nonzero_mask]
    return snb


def _r_squared(y_true, y_pred):
    y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred)**2)  
    ss_tot = np.sum((y_true - np.mean(y_true))**2)  
    r2 = 1 - (ss_res / ss_tot)
    return r2 

def _estimate_params_single(
    center: np.ndarray,
    image: np.ndarray,
    window: int,
    refine_centers: bool,
    verbose: bool,
) -> Union[FitParams2D, FitParams3D]:
    
    if image.ndim == 2:
        return _estimate_params_single2(center, image, window, refine_centers, verbose)
    elif image.ndim == 3:
        return _estimate_params_single3(center, image, window, refine_centers, verbose)
    else:
        raise ValueError("Image must have 2 or 3 dimensions")

def _estimate_params_single2(
    center: np.ndarray,
    image: np.ndarray,
    window: int,
    refine_centers: bool,
    verbose: bool,
) -> FitParams2D:
    x_range = np.arange(-window, window + 1)
    y_range = np.arange(-window, window + 1)
    y, x = np.meshgrid(y_range, x_range, indexing="ij")

    # Crop around the spot with interpolation
    y_indices, x_indices = np.mgrid[
        center[0] - window : center[0] + window + 1,
        center[1] - window : center[1] + window + 1
    ]
    region = map_coordinates(image, [y_indices, x_indices], order=3, mode='reflect')

    try:
        mi, ma = np.min(region), np.max(region)
        region = (region - mi) / (ma - mi)

        initial_guess = (0, 0, 1.5, 1, 0)  # y0, x0, sigma, A, B

        if refine_centers:
            lower_bounds = (-1, -1, 0.1, 0.5, -0.5)  # y0, x0, sigma, A, B
            upper_bounds = (1, 1, 10, 1.5, 0.5)  # y0, x0, sigma, A, B
        else:
            lower_bounds = (-1e-6, -1e-6, 0.1, 0.5, -0.5)
            upper_bounds = (1e-6, 1e-6, 10, 1.5, 0.5)

        popt, _ = curve_fit(
            _gaussian_2d,
            (y.ravel(), x.ravel()),
            region.ravel(),
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
        )
        
        pred = _gaussian_2d((y.ravel(), x.ravel()), *popt)  
        r_squared = _r_squared(region.ravel(), pred)

    except Exception as _:
        if verbose:
            log.warning("Gaussian fit failed. Returning NaN")
        mi, ma = np.nan, np.nan
        popt = np.full(5, np.nan)
        r_squared = 0

    return FitParams2D(
        fwhm=FWHM_CONSTANT * popt[2],
        offset_y=popt[0],
        offset_x=popt[1],
        intens_A=(popt[3]+popt[4])*(ma - mi),
        intens_B=popt[4] * (ma - mi) + mi,
        r_squared=r_squared
    )

def _estimate_params_single3(
    center: np.ndarray,
    image: np.ndarray,
    window: int,
    refine_centers: bool,
    verbose: bool,
) -> FitParams3D:
    z_range = np.arange(-window, window + 1)
    y_range = np.arange(-window, window + 1)
    x_range = np.arange(-window, window + 1)
    z, y, x = np.meshgrid(z_range, y_range, x_range, indexing="ij")

    # Crop around the spot with interpolation
    z_indices, y_indices, x_indices = np.mgrid[
        center[0] - window : center[0] + window + 1,
        center[1] - window : center[1] + window + 1,
        center[2] - window : center[2] + window + 1
    ]
    region = map_coordinates(image, [z_indices, y_indices, x_indices], order=3, mode='reflect')

    try:
        mi, ma = np.min(region), np.max(region)
        region = (region - mi) / (ma - mi)

        initial_guess = (0, 0, 0, 1.5, 2., 1, 0)  # z0, y0, x0, sigmayx, sigmaz, A, B

        if refine_centers:
            lower_bounds = (-1, -1, -1, 0.1, 0.1, 0.5, -0.5)
            upper_bounds = (1, 1, 1, 10, 10, 1.5, 0.5)
        else:
            lower_bounds = (-1e-6, -1e-6, -1e-6, 0.1, 0.1, 0.5, -0.5)
            upper_bounds = (1e-6, 1e-6, 1e-6, 10, 10, 1.5, 0.5)

        popt, _ = curve_fit(
            _gaussian_3d,
            (z.ravel(), y.ravel(), x.ravel()),
            region.ravel(),
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
        )

        pred = _gaussian_3d((z.ravel(), y.ravel(), x.ravel()), *popt)
        r_squared = _r_squared(region.ravel(), pred)

    except Exception as _:
        if verbose:
            log.warning("3D Gaussian fit failed. Returning NaN")
        mi, ma = np.nan, np.nan
        popt = np.full(7, np.nan)
        r_squared = 0

    return FitParams3D(
        fwhm_z=FWHM_CONSTANT * popt[3],
        fwhm_yx=FWHM_CONSTANT * popt[4],
        offset_z=popt[0],
        offset_y=popt[1],
        offset_x=popt[2],
        intens_A=(popt[5]+popt[6])*(ma - mi),
        intens_B=popt[6] * (ma - mi) + mi,
        r_squared=r_squared
    )

def estimate_params(
    img: np.ndarray,
    centers: np.ndarray,
    window: int = 5,
    max_workers: int = 1,
    refine_centers: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """Estimates Gaussian parameters of all spots in an image (given by centers) of the image by fitting a 2D Gaussian
       centered at each spot.

    Args:
        img (np.ndarray): input image
        centers (np.ndarray): centers of the spots to fit Gaussians on
        window (int, optional): window size around the spot to consider during the fitting. Defaults to 5.
        max_workers (int, optional): number of workers for parallelization. Defaults to 1.

    Returns:
        params: SpotParams with the following attributes:
            fwhm (np.ndarray): FWHM of the spots
            offset (np.ndarray): offset of the spots
            peak_range (np.ndarray): peak range of the spots
    """
    img = np.pad(img, window, mode="reflect")
    centers = np.asarray(centers) + window
    if max_workers == 1:
        params = tuple(
            _estimate_params_single(
                p,
                image=img,
                window=window,
                refine_centers=refine_centers,
                verbose=verbose,
            )
            for p in tqdm(centers, desc="Spots fitting", disable=not verbose)
        )
    else:
        _partial_estimate_params_single = partial(
            _estimate_params_single,
            image=img,
            window=window,
            refine_centers=refine_centers,
            verbose=verbose,
        )
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            params = tuple(
                tqdm(
                    executor.map(_partial_estimate_params_single, centers),
                    total=len(centers),
                    desc="Estimating FWHM of spots",
                    disable=not verbose,
                )
            )

    if img.ndim == 2:
        keys = tuple(f.name for f in fields(FitParams2D))
        params = FitParams2D(
            **dict((k, np.array([getattr(p, k) for p in params])) for k in keys)
        )
    elif img.ndim == 3:
        keys = tuple(f.name for f in fields(FitParams3D))
        params = FitParams3D(
            **dict((k, np.array([getattr(p, k) for p in params])) for k in keys)
        )
    else:
        raise ValueError("Image must have 2 or 3 dimensions")
    
    return params
