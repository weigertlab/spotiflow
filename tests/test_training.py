import numpy as np
from spotipy_torch.utils import points_to_prob
from spotipy_torch.data import SpotsDataset
from spotipy_torch.model import SpotipyModelConfig, SpotipyTrainingConfig, Spotipy
import lightning.pytorch as pl
from numerize.numerize import numerize
import torch


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

    device = "cuda" if torch.cuda.is_available() else "mps"

    X, P = example_data(64)
    Xv, Pv = example_data(4)

    backbone = "unet"
    batch_norm = False
    batch_norm = True

    compute_flow = True
    # compute_flow = False

    n_levels = 4

    config = SpotipyModelConfig(
        backbone=backbone,
        levels=n_levels,
        compute_flow=compute_flow,
        fmap_inc_factor=2,
        background_remover=False,
        batch_norm=batch_norm,
    )
    train_config = SpotipyTrainingConfig(num_epochs=20)

    model = Spotipy(config)

    print(f"Total params: {numerize(sum(p.numel() for p in model.parameters()))}")

    data = SpotsDataset(
        X, P, compute_flow=compute_flow, downsample_factors=(1, 2, 4, 8)[:n_levels]
    )
    data_v = SpotsDataset(
        Xv, Pv, compute_flow=compute_flow, downsample_factors=(1, 2, 4, 8)[:n_levels]
    )

    logger = pl.loggers.TensorBoardLogger(
        save_dir="foo",
        name=f"{backbone}_batch_norm_{batch_norm}_flow_{compute_flow}",
    )

    model.fit(data, data_v, train_config, device, logger=logger, deterministic=False)
