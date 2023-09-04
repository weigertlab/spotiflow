import numpy as np
from spotipy_torch.utils import points_to_prob
from spotipy_torch.data import SpotsDataset
from spotipy_torch.model import SpotipyModelConfig, SpotipyTrainingConfig, Spotipy
import lightning.pytorch as pl
import torch

device = "cuda" if torch.cuda.is_available() else "mps"


def example_data(n_samples: int = 64, size: int = 256):
    def _single():
        p = np.random.randint(0, 200, (20, 2))
        x = points_to_prob(p, (256, 256))
        x = x + 0.2 * np.random.normal(0, 1, x.shape).astype(np.float32)
        return x, p

    X, P = tuple(zip(*tuple(_single() for _ in range(n_samples))))
    X, P = np.stack(X), np.stack(P)
    return X, P


if __name__ == "__main__":

    X, P = example_data(64)
    Xv, Pv = example_data(4)

    config = SpotipyModelConfig(
        compute_flow=True, background_remover=False, batch_norm=False
    )
    train_config = SpotipyTrainingConfig()

    model = Spotipy(config)

    data = SpotsDataset(X, P, compute_flow=True, downsample_factors=(1, 2, 4, 8))
    data_v = SpotsDataset(Xv, Pv, compute_flow=True, downsample_factors=(1, 2, 4, 8))

    logger = pl.loggers.TensorBoardLogger(
        save_dir="foo",
        name="foo",
    )

    model.fit(data, data_v, train_config, device, logger=logger)
