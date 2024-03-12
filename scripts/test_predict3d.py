from spotiflow.model import Spotiflow
import numpy as np
import tifffile
from spotiflow.augmentations.transforms3d import Crop3D
from spotiflow.utils import read_coords_csv3d, points_matching
import torch

if __name__ == "__main__":
    model = Spotiflow.from_folder("/data/tmp/spotiflow_3d_debug/synth3d")
    arr = tifffile.imread("/data/spots/datasets_3d/synth3d/test/synth_00065_snr_16_density_0.0877_aberr_0.02.tif")
    gt = read_coords_csv3d("/data/spots/datasets_3d/synth3d/test/synth_00065_snr_16_density_0.0877_aberr_0.02.csv")
    # crop = Crop3D((8, 128, 128), probability=1.0, point_priority=0.5)
    # arr, gt = crop.apply(torch.from_numpy(arr), torch.from_numpy(gt))
    # spots, details = model.predict(arr, subpix=True)
    spots, details = model.predict(arr, subpix=True)
    stats = points_matching(gt, spots)
    print(stats.f1)
    tifffile.imwrite(
        "/data/tmp/spotiflow_3d_debug/synth_00065_snr_16_density_0.0877_aberr_0.02.tif",
        arr,
    )
    print(spots)
    np.savetxt(
        "/data/tmp/spotiflow_3d_debug/synth_00065_snr_16_density_0.0877_aberr_0.02_pred.csv",
        spots,
        delimiter=",",
        header="axis-0,axis-1,axis-2",
    )
