"""
Wrapper for Spotiflow to use it as a starfish spot detector replacement.
Several mock classes are created when starfish is not installed or not working properly.
In that case, a RuntimeError is raised when trying to use the class SpotiflowDetector, but it does not 
break at import time to allow the user to use spotiflow without having to install starfish.
"""
import logging
import sys
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ..model import Spotiflow
from ..utils import NotRegisteredError, normalize

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

try:
    from starfish.core.imagestack.imagestack import ImageStack
    from starfish.core.spots.FindSpots._base import FindSpotsAlgorithm
    from starfish.core.types import (Axes, Features, PerImageSliceSpotResults,
                                     SpotAttributes, SpotFindingResults)
    from xarray.core.coordinates import DataArrayCoordinates
    STARFISH_IE = False
except ImportError as ie:
    # Starfish is not installed, but not big deal. We mock several classes so that the code works without starfish
    # but raises an exception if the user tries to use it while maintaining typing coherence.

    STARFISH_IE = True
    class FindSpotsAlgorithm:
        """Mock class for starfish.core.spots.FindSpots._base.FindSpotsAlgorithm
        """
        def __init__(self, *args, **kwargs):
            raise RuntimeError("starfish is not installed or its installation is not working as expected. Please install starfish to use the class SpotiflowDetector. If starfish is already installed, please check that the installation is working properly.")
    
    class PerImageSliceSpotResults:
        """Mock class for starfish.core.types.PerImageSliceSpotResults
        """
        def __init__(self, *args, **kwargs):
            raise RuntimeError("starfish is not installed or its installation is not working as expected. Please install starfish to use the class SpotiflowDetector. If starfish is already installed, please check that the installation is working properly.")
    
    class ImageStack:
        """Mock class for starfish.core.imagestack.imagestack.ImageStack
        """
        def __init__(self, *args, **kwargs):
            raise RuntimeError("starfish is not installed or its installation is not working as expected. Please install starfish to use the class SpotiflowDetector. If starfish is already installed, please check that the installation is working properly.")
    
    class SpotFindingResults:
        """Mock class for starfish.core.types.SpotFindingResults
        """
        def __init__(self, *args, **kwargs):
            raise RuntimeError("starfish is not installed or its installation is not working as expected. Please install starfish to use the class SpotiflowDetector. If starfish is already installed, please check that the installation is working properly.")

class SpotiflowDetector(FindSpotsAlgorithm):
    """Wrapper for Spotiflow to use it as a starfish spot detector replacement.
    """
    def __init__(
            self,
            model: Union[Spotiflow, str, Path],
            prob_threshold: float = None,
            n_tiles: Union[Tuple[int], None] = None,
            min_distance: int = 2,
            max_tile_size: int = 4096,
            measurement_type='max',
            is_volume: bool = False,
            crop_to_shape: Optional[Tuple[int]] = None,
            normalizer: Optional[Callable] = lambda img: normalize(img, 1, 99.8),
    ) -> None:
        if STARFISH_IE: # Starfish was not imported properly
            raise RuntimeError("starfish is not installed or its installation is not working as expected. Please install starfish to use the class SpotiflowDetector. If starfish is already installed, please check that the installation is working properly.")
        # Starfish stuff
        self.is_volume = is_volume
        assert not self.is_volume, "Spotiflow currently only works in 2D stacks"
        self.measurement_function = self._get_measurement_function(measurement_type)

        # Spotipy args
        self.model = model
        if isinstance(self.model, Spotiflow):
            try:
                self.model = torch.compile(self.model)
            except RuntimeError as e:
                log.warn("Could not compile the model. Will proceed without torch compilation.")
        self.n_tiles = n_tiles
        self.min_distance = min_distance
        self.probability_threshold = prob_threshold
        self.model = None
        self._max_tile_size = max_tile_size
        self._normalizer = normalizer
        self._crop_to_shape = crop_to_shape

    @property
    def normalizer(self) -> Callable:
        return self._normalizer

    def _populate_model(self):
        if isinstance(self.model, str):
            try:
                self.model = Spotiflow.from_pretrained(self.model)
            except NotRegisteredError:
                self.model = Spotiflow.from_folder(self.model)
        elif isinstance(self.model, Path):
            self.model = Spotiflow.from_folder(self.model)
        
        try:
            self.model = torch.compile(self.model)
        except RuntimeError as e:
            log.warn("Could not compile the model. Will proceed without torch compilation.")
        
        if isinstance(self.model, Spotiflow):
            return self.model
        else:
            raise ValueError(f"model argument for initializing must be a path to a model, a Spotiflow object, or a registered model name")

    def image_to_spots(
            self, data_image: np.ndarray,
    ) -> PerImageSliceSpotResults:
        """
        Find spots using a Spotiflow model 
        Parameters
        ----------
        data_image : np.ndarray
            image containing spots to be detected
        Returns
        -------
        PerImageSpotResults :
            includes a SpotAttributes DataFrame of metadata containing the coordinates, intensity
            and radius of each spot, as well as any extra information collected during spot finding.
        """

        self._populate_model()
        data_image = np.asarray(data_image)

        if self._crop_to_shape is not None:
            data_image = data_image[:self._crop_to_shape[0], :self._crop_to_shape[1]]

        n_tiles = self.n_tiles if self.n_tiles is not None \
                               else tuple(max(1,int(np.ceil(s/self._max_tile_size))) for s in data_image.shape)


        img = data_image.reshape(data_image) if data_image.ndim > 2 else data_image

        spotipy_kwargs = {
            "prob_thresh": self.probability_threshold,
            "min_distance": self.min_distance,
            "n_tiles": n_tiles,
            "verbose": False,
            "peak_mode": "fast",
            "subpix": True if self.model.config.compute_flow else False,
            "normalizer": self.normalizer,
        }

        detections, details = self.model.predict(
                img,
                **spotipy_kwargs
        )


        if detections.shape[0] == 0:
            empty_spot_attrs = SpotAttributes.empty(
                extra_fields=[Features.INTENSITY, Features.SPOT_ID])
            return PerImageSliceSpotResults(spot_attrs=empty_spot_attrs, extras=None)

        # Measure intensities
        if self.is_volume:
            raise NotImplementedError
        else:
            z_inds = np.asarray([0 for _ in range(detections.shape[0])])
            y_inds = detections[:, 0]
            x_inds = detections[:, 1]
            # ~ radius as in classical LoG
            radius = np.asarray([np.sqrt(2)*self.model.config.sigma for _ in range(detections.shape[0])])
            intensities = details.prob

        # construct dataframe
        spot_data = pd.DataFrame(
            data={
                Features.INTENSITY: intensities,
                Axes.ZPLANE.value: z_inds,
                Axes.Y.value: y_inds,
                Axes.X.value: x_inds,
                Features.SPOT_RADIUS: radius,
            }
        )
        spots = SpotAttributes(spot_data)
        spots.data[Features.SPOT_ID] = np.arange(spots.data.shape[0])
        return PerImageSliceSpotResults(spot_attrs=spots, extras=None)

    def run(
            self,
            image_stack: ImageStack,
            reference_image: Optional[ImageStack] = None,
            n_processes: Optional[int] = None,
            *args,
    ) -> SpotFindingResults:
        """
        Find spots in the given ImageStack using Spotiflow.
        If a reference image is provided the spots will be detected there then measured
        across all rounds and channels in the corresponding ImageStack. If a reference_image
        is not provided spots will be detected independently in each channel. 
        Parameters
        ----------
        image_stack : ImageStack
            ImageStack where we find the spots in.
        reference_image : Optional[ImageStack]
            A reference image. Maintained for compatibility with starfish's API. Will raise an error if provided.
        n_processes : Optional[int] = None,
            Maintained for compatibility with starfish's API, but not used.
        """
        spot_finding_method = partial(self.image_to_spots, *args)
        if reference_image:
            raise NotImplementedError("A reference image is not supported by Spotiflow. Please provide None as reference_image.")

        spot_attributes_list = []
        for r in tqdm(range(image_stack.shape[Axes.ROUND]), desc="Processing cycles"):
            for ch in range(image_stack.shape[Axes.CH]):
                # img = np.squeeze(image_stack.sel({Axes.ROUND:r, Axes.CH:ch, Axes.ZPLANE:0}).xarray.data)
                # avoids internal deepcopy and thus blowing up memory for large images
                img = image_stack._tile_data.get_tile(r=r, ch=ch, z=0).numpy_array
                res = spot_finding_method(img)
                spot_attributes_list.append((res, {Axes.ROUND:r, Axes.CH:ch, Axes.ZPLANE:0}))

        # the default way to get coords used by starfish is
        # imagestack_coords=image_stack.xarray.coords
        # which results in a copy of the data
        # this avoids it
        imagestack_coords = DataArrayCoordinates(image_stack._data)

        results = SpotFindingResults(imagestack_coords=imagestack_coords,
                                        log=image_stack.log,
                                        spot_attributes_list=spot_attributes_list)
        return results