from spotiflow.model import SpotiflowModelConfig, Spotiflow
from numerize.numerize import numerize
import torch
from utils import example_data


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "mps"

    X, P = example_data(64, sigma=3, noise=0.01)
    Xv, Pv = example_data(4, sigma=3, noise=0.01)

    config = SpotiflowModelConfig()

    model = Spotiflow(config)

    print(f"Total params: {numerize(sum(p.numel() for p in model.parameters()))}")

    model.fit(X, P, Xv, Pv, save_dir="tmp", train_config={"num_epochs": 10})
