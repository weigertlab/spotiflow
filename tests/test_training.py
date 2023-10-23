from spotipy_torch.data import SpotsDataset
from spotipy_torch.model import SpotipyModelConfig, SpotipyTrainingConfig, Spotipy
import lightning.pytorch as pl
from numerize.numerize import numerize
import torch
from utils import example_data


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "mps"

    X, P = example_data(64)
    Xv, Pv = example_data(4)

    backbone = "unet"
    batch_norm = False
    batch_norm = True

    compute_flow = True
    # compute_flow = False

    n_levels = 2

    config = SpotipyModelConfig(
        backbone=backbone,
        levels=n_levels,
        compute_flow=compute_flow,
        mode="slim",
        fmap_inc_factor=2,
        background_remover=False,
        batch_norm=batch_norm,
    )

    train_config = SpotipyTrainingConfig(num_epochs=20, sigma=1.5)

    data = SpotsDataset(
        X, P, compute_flow=compute_flow, downsample_factors=(1, 2, 4, 8)[:n_levels]
    )
    data_v = SpotsDataset(
        Xv, Pv, compute_flow=compute_flow, downsample_factors=(1, 2, 4, 8)[:n_levels]
    )

    model = Spotipy(config)

    print(f"Total params: {numerize(sum(p.numel() for p in model.parameters()))}")

    logger = pl.loggers.TensorBoardLogger(
        save_dir="foo",
        name=f"{backbone}_batch_norm_{batch_norm}_flow_{compute_flow}",
    )

    model.fit(data, data_v, train_config, device, logger=logger, deterministic=False)

    # model = Spotipy.from_pretrained(f"foo", map_location=device)
    # model.to(device)
    # p1, details1 = model.predict(data._images[0], device=device, subpix=False)
    # p2, details2 = model.predict(data._images[0], device=device, subpix=True)
