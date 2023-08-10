from spotipy_torch import datasets
from spotipy_torch import utils
from spotipy_torch.model import Spotipy

from pathlib import Path
import argparse
import torch





parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="/data/spots/datasets/telomeres/test")
parser.add_argument("--model-dir", type=str, default="/data/spots/results/telomeres/spotipy_torch_v2")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--cutoff-distance", type=float, default=3.)
args = parser.parse_args()


model = Spotipy(pretrained_path=args.model_dir,
                inference_mode=True,
                which="best",
                device="cuda" if torch.cuda.is_available() else "cpu")

print("Loading data...")
ds = datasets.AnnotatedSpotsDataset(Path(args.data_dir),
                                       downsample_factors=[2**lv for lv in range(model._levels)],
                                       sigma=1, 
                                       mode="max",
                                       augment_probability=0,
                                       use_gpu=False,
                                       size=None,
                                       norm_percentiles=(1, 99.8))


print("Predicting...")
preds = model.predict_dataset(
    ds,
    batch_size=args.batch_size,
)

stat = utils.points_matching_dataset(
    ds.get_centers(),
    preds,
    args.cutoff_distance,
    by_image=True
)

print(stat)