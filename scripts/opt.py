from spotipy_torch import datasets
from spotipy_torch.model import Spotipy

from pathlib import Path
import argparse
import torch





parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="/data/spots/datasets/telomeres")
parser.add_argument("--model-dir", type=str, default="/data/spots/results/telomeres/spotipy_torch_v2")
parser.add_argument("--batch-size", type=int, default=4)
args = parser.parse_args()


print("Loading model...")
model = Spotipy(pretrained_path=args.model_dir,
                inference_mode=True,
                which="best",
                device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


model = torch.compile(model)


print("Loading data...")
val_ds = datasets.AnnotatedSpotsDataset(Path(args.data_dir)/"val",
                                       downsample_factors=[2**lv for lv in range(model._levels)],
                                       sigma=1, 
                                       mode="max",
                                       augment_probability=0,
                                       use_gpu=False,
                                       size=None,
                                       norm_percentiles=(1, 99.8))


# Train model
print("Optimizing threshold...")
model.optimize_threshold(val_ds, min_distance=1)
