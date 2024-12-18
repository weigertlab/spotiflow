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

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

try:
    from starfish.core.image.Filter.util import determine_axes_to_group_by
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
    def __init__(
            self,
            model: Union[Spotiflow, str, Path],
            probability_threshold: float = None,
            n_tiles: Union[Tuple[int], None] = None,
            min_distance: int = 2,
            measurement_type='max',
            subpix: bool = False,
            is_volume: bool = False,
            pretrained_model: Optional[str] = None,
    ) -> None:
        self.is_volume = is_volume
        self.measurement_function = self._get_measurement_function(measurement_type)

        # Spotiflow args
        self.pretrained_model = pretrained_model
        self.n_tiles = n_tiles
        self.min_distance = min_distance
        self.probability_threshold = probability_threshold
        self.model = model
        assert not subpix, "Subpixel localization is not supported in the Starfish integration yet"

    def _populate_model(self):
        if isinstance(self.model, str):
            try:
                self.model = Spotiflow.from_pretrained(self.model)
            except NotRegisteredError:
                self.model = Spotiflow.from_folder(self.model)
        elif isinstance(self.model, Path):
            self.model = Spotiflow.from_folder(self.model)
        
        if isinstance(self.model, Spotiflow):
            return self.model
        else:
            raise ValueError(f"model argument for initializing must be a path to a model, a Spotiflow object, or a registered model name")

    def image_to_spots(
            self, data_image: np.ndarray,
    ) -> PerImageSliceSpotResults:
        """
        Find spots using Spotiflow

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

        spotiflow_kwargs = {
            "prob_thresh": self.probability_threshold,
            "min_distance": self.min_distance,
            "n_tiles": self.n_tiles,
            "verbose": False,
            "peak_mode": "fast",
            "subpix": True if self.model.config.compute_flow else False,
        }
        detections, details = self.model.predict(
                data_image,
                **spotiflow_kwargs
        )

        if detections.shape[0] == 0:
            empty_spot_attrs = SpotAttributes.empty(
                extra_fields=[Features.INTENSITY, Features.SPOT_ID])
            return PerImageSliceSpotResults(spot_attrs=empty_spot_attrs, extras=None)

        # Measure intensities
        if not self.is_volume:
            z_inds = np.asarray([0 for _ in range(detections.shape[0])])
            y_inds = detections[:, 0].astype(int)
            x_inds = detections[:, 1].astype(int)
        else:
            z_inds = detections[:, 0].astype(int)
            y_inds = detections[:, 1].astype(int)
            x_inds = detections[:, 2].astype(int)

        # TODO: use estimated fwhm 
        radius = np.asarray([np.sqrt(2 if not self.is_volume else 3) for _ in range(detections.shape[0])])
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
        is not provided spots will be detected _independently_ in each channel. This assumes
        a non-multiplex imaging experiment, as only one (ch, round) will be measured for each spot.

        Parameters
        ----------
        image_stack : ImageStack
            ImageStack where we find the spots in.
        reference_image : Optional[ImageStack]
            (Optional) a reference image. If provided, spots will be found in this image, and then
            the locations that correspond to these spots will be measured across each channel.
        n_processes : Optional[int] = None,
            Number of processes to devote to spot finding.
        """
        spot_finding_method = partial(self.image_to_spots, *args)
        if reference_image:
            data_image = reference_image._squeezed_numpy(*{Axes.ROUND, Axes.CH})
            reference_spots = spot_finding_method(data_image)
            results = spot_finding_utils.measure_intensities_at_spot_locations_across_imagestack(
                data_image=image_stack,
                reference_spots=reference_spots,
                measurement_function=self.measurement_function)
        else:
            spot_attributes_list = image_stack.transform(
                func=spot_finding_method,
                group_by=determine_axes_to_group_by(self.is_volume),
                n_processes=n_processes
            )

            # If not a volume, merge spots from the same round/channel but different z slices
            if not self.is_volume:
                merged_z_tables = defaultdict(pd.DataFrame)  # type: ignore
                for i in range(len(spot_attributes_list)):
                    spot_attributes_list[i][0].spot_attrs.data['z'] = spot_attributes_list[i][1]['z']
                    r = spot_attributes_list[i][1][Axes.ROUND]
                    ch = spot_attributes_list[i][1][Axes.CH]
                    merged_z_tables[(r, ch)] = merged_z_tables[(r, ch)].append(
                        spot_attributes_list[i][0].spot_attrs.data)
                new = []
                r_chs = sorted([*merged_z_tables])
                selectors = list(image_stack._iter_axes({Axes.ROUND, Axes.CH}))
                for i, (r, ch) in enumerate(r_chs):
                    merged_z_tables[(r, ch)]['spot_id'] = range(len(merged_z_tables[(r, ch)]))
                    spot_attrs = SpotAttributes(merged_z_tables[(r, ch)].reset_index(drop=True))
                    new.append((PerImageSliceSpotResults(spot_attrs=spot_attrs, extras=None),
                               selectors[i]))

                spot_attributes_list = new

            results = SpotFindingResults(imagestack_coords=image_stack.xarray.coords,
                                         log=image_stack.log,
                                         spot_attributes_list=spot_attributes_list)
        return results
