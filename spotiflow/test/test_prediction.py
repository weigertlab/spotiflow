import numpy as np
import pytest
import torch

from spotiflow.model import Spotiflow, SpotiflowModelConfig
from typing import Tuple


AVAILABLE_DEVICES = [None, "auto", "cpu"]
if torch.cuda.is_available():
    AVAILABLE_DEVICES += ["cuda"]
if torch.backends.mps.is_available():
    AVAILABLE_DEVICES += ["mps"]

AVAILABLE_DEVICES = tuple(AVAILABLE_DEVICES)

np.random.seed(42)
torch.random.manual_seed(42)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


@pytest.mark.parametrize("img_size", ((64, 64), (101, 241)))
@pytest.mark.parametrize("n_tiles", ((1,1), (2,2), (2,2,1)))
@pytest.mark.parametrize("scale", (1/3, 1., 2.))
@pytest.mark.parametrize("in_channels", (1, 3, 5))
@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
@pytest.mark.parametrize("subpix", (True, False))
def test_predict(img_size: Tuple[int, int],
                 n_tiles: Tuple[int, ...],
                 scale: float,
                 in_channels: int,
                 device: str,
                 subpix: bool,
                 ):
    img = np.random.randn(*img_size, in_channels).astype(np.float32)
    model_config = SpotiflowModelConfig(
        levels=2,
        in_channels=in_channels,
        out_channels=1,
    )
    model = Spotiflow(model_config)
    orig_device = str(next(model.parameters()).device)

    wrong_scale, not_implemented = False, False
    if scale < 1:
        inv_scale = int(1/scale)
        wrong_scale = any(s % inv_scale != 0 for s in img_size)

    if scale != 1 and subpix:
        not_implemented = True


    if not wrong_scale and not not_implemented:
        pred, details = model.predict(
            img,
            n_tiles=n_tiles,
            scale=scale,
            verbose=False,
            device=device,
            subpix=subpix,
        )
        if "cuda" in AVAILABLE_DEVICES and device in ("auto", "cuda"):
            assert str(next(model.parameters()).device).startswith("cuda")
        elif "mps" in AVAILABLE_DEVICES and device in ("auto", "mps"):
            assert str(next(model.parameters()).device).startswith("mps")
        elif device is None:
            assert str(next(model.parameters()).device).startswith(orig_device)
        else:
            assert str(next(model.parameters()).device).startswith("cpu")
        assert all(p==s for p, s in zip(details.heatmap.shape, img_size)), f"Wrong heatmap shape: expected {img_size}, got {details.heatmap.shape}"
        if pred.shape[0] > 0:
            assert pred.min() >= 0, "Point detection coordinates should be non-negative"
            assert pred.max() < img_size[0] or pred.max() < img_size[1], "Point detection coordinates should be within the image dimensions"
    elif wrong_scale and not not_implemented:
        with pytest.raises(AssertionError):
            pred, details = model.predict(
                img,
                n_tiles=n_tiles,
                scale=scale,
                verbose=False,
                device=device,
                subpix=subpix,
            )
    elif not_implemented:
        with pytest.raises(NotImplementedError):
            pred, details = model.predict(
                img,
                n_tiles=n_tiles,
                scale=scale,
                verbose=False,
                device=device,
                subpix=subpix,
            )


if __name__ == "__main__":
    test_predict(
        img_size=(64, 64),
        n_tiles=(2, 2, 1),
        scale=1/3,
        in_channels=3,
        device=None,
        subpix=True,
    )

