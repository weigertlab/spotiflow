from spotipy_torch.data import SpotsDataset
from spotipy_torch.model import Spotipy
from spotipy_torch.utils import normalize

from pathlib import Path
import argparse
import torch





parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="/data/spots/datasets/telomeres")
parser.add_argument("--model-dir", type=str, default="/data/spots/results/telomeres/spotipy_torch_v2")
parser.add_argument("--which", type=str, default="best")
parser.add_argument("--batch-size", type=int, default=1)
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading model...")
model = Spotipy.from_pretrained(
    pretrained_path=args.model_dir,
    inference_mode=True,
).to(torch.device(device))

model = torch.compile(model)


print("Loading data...")
val_ds = SpotsDataset.from_folder(Path(args.data_dir)/"val",
                                       augmenter=None,
                                       downsample_factors=[2**lv for lv in range(model._levels)],
                                       sigma=1, 
                                       mode="max",
                                       normalizer= lambda x: normalize(x, 1, 99.8))


# Train model
print("Optimizing threshold...")
model.optimize_threshold(val_ds, min_distance=1, batch_size=args.batch_size, device=device)

print("Re-saving model config...")
model.save(args.model_dir, which=args.which, only_config=True)
