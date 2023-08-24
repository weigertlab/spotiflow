from spotipy_torch.data import SpotsDataset
from spotipy_torch import utils
from spotipy_torch.model import Spotipy
from tqdm.auto import tqdm

from pathlib import Path
import argparse
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="/data/spots/datasets/telomeres")
parser.add_argument("--model-dir", type=str, default="/data/spots/results/telomeres/spotipy_torch_v2")
parser.add_argument("--out-dir", type=str, default="/data/spots/results/telomeres/spotipy_torch_v2")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--opt-split", type=str, default=None)
parser.add_argument("--pred-split", type=str, default="test")
parser.add_argument("--cutoff-distance", type=float, default=3.)
parser.add_argument("--which", type=str, choices=["best", "last"], default="last")
parser.add_argument("--threshold", type=float, required=False, default=None)
args = parser.parse_args()


model = Spotipy(pretrained_path=args.model_dir,
                inference_mode=True,
                which=args.which,
                device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

model = torch.compile(model)

if args.opt_split is not None:
    print(f"Optimizing and updating threshold on {args.opt_split}...")
    opt_ds = SpotsDataset.from_folder(
        path=Path(args.data_dir)/args.opt_split,
        augmenter=None,
        downsample_factors=[2**lv for lv in range(model._levels)],
        sigma=1.,
        mode="max",
        normalizer=lambda img: utils.normalize(img, 1, 99.8),
    )
    model.optimize_threshold(opt_ds, min_distance=1, batch_size=1)
    model._save_model(args.model_dir, which=args.which, only_config=True)


print("Loading data...")
pred_ds = SpotsDataset.from_folder(
    path=Path(args.data_dir)/args.pred_split,
    augmenter=None,
    downsample_factors=[2**lv for lv in range(model._levels)],
    sigma=1.,
    mode="max",
    normalizer=lambda img: utils.normalize(img, 1, 99.8),
)

out_dir_split = Path(args.out_dir)/args.pred_split
out_dir_split.mkdir(exist_ok=True, parents=True)

fnames = pred_ds.image_files

for i, fname in tqdm(enumerate(fnames), desc="Predicting and writing", total=len(fnames)):
    normalized_img = pred_ds._images[i]
    pts, _ = model.predict(normalized_img, prob_thresh=args.threshold, min_distance=1, verbose=False)
    utils.write_coords_csv(pts, out_dir_split/f"{fname}.csv")
