import numpy as np
import pytest
import torch

from spotiflow.model import Spotiflow, SpotiflowModelConfig
from tempfile import TemporaryDirectory


def equal_states_dict(state_dict_1, state_dict_2):
    if len(state_dict_1) != len(state_dict_2):
        return False

    for ((k_1, v_1), (k_2, v_2)) in zip(
        state_dict_1.items(), state_dict_2.items()
    ):
        if k_1 != k_2:
            return False
        
        if not torch.allclose(v_1, v_2):
            return False
    return True


DEVICE_STR = "cpu"

np.random.seed(42)
torch.random.manual_seed(42)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


@pytest.mark.parametrize("in_channels", (1, 3))
def test_save_load(
                   in_channels: int,
                  ):
    model_config = SpotiflowModelConfig(
        levels=2,
        in_channels=in_channels,
        out_channels=1,
        background_remover=in_channels==1,
    )
    model = Spotiflow(model_config)

    with TemporaryDirectory() as tmpdir:
        model.save(tmpdir, which="best", update_thresholds=True)
        model_same = Spotiflow.from_folder(tmpdir, map_location=DEVICE_STR)
        with pytest.raises(AssertionError):
            _ = Spotiflow.from_folder(tmpdir, map_location=DEVICE_STR, which="notexist")

    assert model_same.config == model.config, "Model configs are not equal"
    assert equal_states_dict(model.state_dict(), model_same.state_dict()), "Model states are not equal"

