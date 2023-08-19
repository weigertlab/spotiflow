from spotipy_torch.model import Spotipy
from spotipy_torch.utils import normalize
from pathlib import Path

import argparse
import tifffile
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", type=str, default="/data/spotipy_torch_v2_experiments/hybiss_spots_v2_unet_lvs4_fmaps32_direct_convperlv3_lossbce_epochs200_lr00003_bsize4_ksize3_sigma1_crop512_posweight10_seed42_tormenter")
parser.add_argument("--img-dir", type=str, default="/data/spots/datasets/hybiss_spots_v2/train/irina_tile1_ch3_tile0_row2617_col4113_rep4.tif")
parser.add_argument("--which", type=str, choices=["best", "last"], default="last")
args = parser.parse_args()

assert Path(args.model_dir).exists(), "Model does not exist!"

model = Spotipy(pretrained_path=args.model_dir,
                inference_mode=True,
                which=args.which,
                device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

assert Path(args.img_dir).exists(), "Image does not exist!"
model = torch.compile(model)
img = tifffile.imread(args.img_dir)


model.predict(
    img,
    n_tiles=(2,2),
    min_distance=1,
    exclude_border=False,
    scale=None,
    peak_mode="skimage",
    normalizer=lambda im: normalize(im, 1, 99.8),
    verbose=True,
)