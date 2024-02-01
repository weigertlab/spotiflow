from spotiflow.data import SpotsDataset
from spotiflow.model import Spotiflow
from spotiflow.utils import normalize

from pathlib import Path
import argparse
import torch





parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="/data/spots/datasets/telomeres")
parser.add_argument("--model-dir", type=str, default="/data/spots/results/telomeres/spotiflow_v2")
parser.add_argument("--which", type=str, default="best")
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--threshold-range", nargs="+", default=(0.3, 0.7))
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading model...")
model = Spotiflow.from_pretrained(
    pretrained_path=args.model_dir,
    inference_mode=True,
).to(torch.device(device))
try:
    model = torch.compile(model)
except RuntimeError:
    print("Could not compile model. Proceeding without torch compilation.")


print("Loading data...")
val_ds = SpotsDataset.from_folder(Path(args.data_dir)/"val",
                                       augmenter=None,
                                       downsample_factors=[2**lv for lv in range(model._levels)],
                                       sigma=1, 
                                       mode="max",
                                       normalizer= lambda x: normalize(x, 1, 99.8))


# Train model
print("Optimizing threshold...")
threshold_range = tuple(float(t) for t in args.threshold_range)
model.optimize_threshold(val_ds, min_distance=1, batch_size=args.batch_size, device=device, threshold_range=threshold_range)

print("Re-saving model config...")
model.save(args.model_dir, which=args.which, update_thresholds=True)
