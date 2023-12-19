from spotiflow.data import SpotsDataset
from spotiflow.model import SpotiflowModelConfig, SpotiflowTrainingConfig, Spotiflow
import lightning.pytorch as pl
from numerize.numerize import numerize
import torch
from utils import example_data


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "mps"

    X, P = example_data(64, sigma=3, noise=0.01)
    Xv, Pv = example_data(4, sigma=3, noise=0.01)

    backbone = "unet"
    batch_norm = False
    batch_norm = True

    compute_flow = True
    # compute_flow = False

    n_levels = 3

    sigma = 1.5

    config = SpotiflowModelConfig(
        backbone=backbone,
        levels=n_levels,
        compute_flow=compute_flow,
        mode="slim",
        sigma=sigma,
        fmap_inc_factor=2,
        background_remover=False,
        batch_norm=batch_norm,
    )

    train_config = SpotiflowTrainingConfig(num_epochs=100, pos_weight=10, batch_size=4)

    data = SpotsDataset(
        X,
        P,
        compute_flow=compute_flow,
        sigma=sigma,
        downsample_factors=(1, 2, 4, 8)[:n_levels],
    )
    data_v = SpotsDataset(
        Xv,
        Pv,
        compute_flow=compute_flow,
        sigma=sigma,
        downsample_factors=(1, 2, 4, 8)[:n_levels],
    )

    model = Spotiflow(config)

    print(f"Total params: {numerize(sum(p.numel() for p in model.parameters()))}")

    logger = pl.loggers.TensorBoardLogger(
        save_dir="foo",
        name=f"{backbone}_batch_norm_{batch_norm}_flow_{compute_flow}",
    )

    model.fit(data, data_v, train_config, device, logger=logger, deterministic=False)
