import logging
import numpy as np
import pytest
import torch

from spotipy_torch.model import Spotipy, SpotipyModelConfig
from typing import Tuple


DEVICE_STR = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

np.random.seed(42)
torch.random.manual_seed(42)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


@pytest.mark.parametrize("img_size", ((64, 64), (128, 128), (101, 241)))
@pytest.mark.parametrize("n_tiles", ((1,1), (2,2), (2,2,1)))
@pytest.mark.parametrize("scale", (1/3, 1., 2.))
@pytest.mark.parametrize("in_channels", (1, 3, 5))
def test_predict(img_size: Tuple[int, int],
                 n_tiles: Tuple[int, ...],
                 scale: float,
                 in_channels: int,
                 ):
    img = np.random.randn(*img_size, in_channels).astype(np.float32)
    model_config = SpotipyModelConfig(
        levels=2,
        in_channels=in_channels,
        out_channels=1,
        background_remover=in_channels==1,
    )
    model = Spotipy(model_config).to(torch.device(DEVICE_STR))
    wrong_scale = False
    if scale < 1:
        inv_scale = int(1/scale)
        wrong_scale = any(s % inv_scale != 0 for s in img_size)

    if not wrong_scale:
        pred, details = model.predict(
            img,
            n_tiles=n_tiles,
            scale=scale,
            verbose=False,
            device=DEVICE_STR
        )
        assert all(p==s for p, s in zip(details.heatmap.shape, img_size)), f"Wrong heatmap shape: expected {img_size}, got {details.heatmap.shape}"
        if pred.shape[0] > 0:
            assert pred.min() >= 0, "Point detection coordinates should be non-negative"
            assert pred.max() < img_size[0] or pred.max() < img_size[1], "Point detection coordinates should be within the image dimensions"
    else:
        with pytest.raises(AssertionError):
            pred, details = model.predict(
                img,
                n_tiles=n_tiles,
                scale=scale,
                verbose=False,
                device=DEVICE_STR
            )


if __name__ == "__main__":
    test_predict(
        img_size=(102, 242),
        n_tiles=(1, 1, 1),
        scale=1/2,
        in_channels=5
    )

