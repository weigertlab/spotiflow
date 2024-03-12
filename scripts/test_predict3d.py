from spotiflow.model import Spotiflow
import numpy as np
import tifffile
from spotiflow.augmentations.transforms3d import Crop3D
from spotiflow.utils import read_coords_csv3d, points_matching, write_coords_csv
import torch

if __name__ == "__main__":
    model = Spotiflow.from_folder("/Users/adomi/tmp/spotiflow_3d/synth3d/model", map_location="cpu")
    arr = tifffile.imread("/Users/adomi/tmp/spotiflow_3d/synth3d/test/synth_00049_snr_14_density_0.0260_aberr_0.02.tif")
    gt = read_coords_csv3d("/Users/adomi/tmp/spotiflow_3d/synth3d/test/synth_00049_snr_14_density_0.0260_aberr_0.02.csv")
    # crop = Crop3D((8, 128, 128), probability=1.0, point_priority=0.5)
    # arr, gt = crop.apply(torch.from_numpy(arr), torch.from_numpy(gt))
    # spots, details = model.predict(arr, subpix=True)
    spots, details = model.predict(arr, subpix=True, device="cpu")
    stats = points_matching(gt, spots)
    print(stats)
    # tifffile.imwrite(
    #     "/data/tmp/spotiflow_3d_debug/synth_00065_snr_16_density_0.0877_aberr_0.02.tif",
    #     arr,
    # )
    # print(spots)
    write_coords_csv(
        spots,
        "/Users/adomi/tmp/spotiflow_3d/synth3d/test_preds/synth_00049_snr_14_density_0.0260_aberr_0.02.csv",
    )
    tifffile.imwrite(
        "/Users/adomi/tmp/spotiflow_3d/synth3d/test_preds/synth_00049_snr_14_density_0.0260_aberr_0.02_heatmap.tif",
        details.heatmap,
    )
