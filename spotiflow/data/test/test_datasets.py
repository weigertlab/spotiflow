import numpy as np
import pytest
from spotiflow.data import Spots3DDataset, SpotsDataset


def _common_assertions(
    item, n_dims: int, downsample: tuple, spatial_dim: int, n_channels: int = 1
):
    assert item["img"].shape == (n_channels,) + (spatial_dim,) * n_dims
    assert item["pts"].shape == (
        10,
        n_dims + (n_dims == 2),
    )  # class label is added only in 2D, needs a full refactor
    assert item["heatmap_lv0"].shape == (1,) + (spatial_dim,) * n_dims
    assert item["heatmap_lv1"].shape == (1,) + (spatial_dim // downsample[1],) * n_dims
    assert item["flow"].shape == (n_dims + 1,) + (spatial_dim,) * n_dims


@pytest.mark.parametrize("n_dims", [2, 3])
@pytest.mark.parametrize("downsample", [(1, 2), (1, 4)])
def test_spots3ddataset_singlechannel(n_dims: int, downsample: tuple):
    _spatial_dim = 16
    _img_dims = (_spatial_dim,) * n_dims
    _n_imgs = 4
    _DatasetCls = SpotsDataset if n_dims == 2 else Spots3DDataset

    imgs = np.random.randn(_n_imgs, *_img_dims).astype(np.float32)
    centers = [np.random.randint(0, _spatial_dim, (10, n_dims)) for _ in range(4)]
    dataset = _DatasetCls(
        imgs,
        centers,
        downsample_factors=downsample,
        normalizer=None,
        add_class_label=n_dims == 2,
        compute_flow=True,
        grid=(1,) * n_dims,
    )
    _item = dataset[0]
    _common_assertions(_item, n_dims, downsample, _spatial_dim, n_channels=1)


@pytest.mark.parametrize("n_dims", [2, 3])
@pytest.mark.parametrize("downsample", [(1, 2), (1, 4)])
def test_spots3ddataset_multichannel(n_dims: int, downsample: tuple):
    _spatial_dim = 16
    _n_channels = 3
    _n_imgs = 4
    _DatasetCls = SpotsDataset if n_dims == 2 else Spots3DDataset

    _img_dims = (_spatial_dim,) * n_dims + (_n_channels,)
    imgs = np.random.randn(_n_imgs, *_img_dims)
    centers = [np.random.randint(0, _spatial_dim, (10, n_dims)) for _ in range(4)]
    dataset = _DatasetCls(
        imgs,
        centers,
        downsample_factors=downsample,
        normalizer=None,
        add_class_label=n_dims == 2,
        compute_flow=True,
        grid=(1,) * n_dims,
    )
    _item = dataset[0]
    _common_assertions(_item, n_dims, downsample, _spatial_dim, _n_channels)
