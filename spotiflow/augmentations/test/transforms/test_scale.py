from numbers import Number

import logging
import pytest
import torch

from spotiflow.augmentations.transforms import IsotropicScale
from spotiflow.augmentations.transforms.utils import _generate_img_from_points
from typing import Tuple

MSE_TOLERANCE = 1e-1


@pytest.mark.parametrize("img_size", [(4, 1, 224, 224), (5, 2, 512, 512), (1, 100, 100), (3, 327, 312)])
@pytest.mark.parametrize("scaling_factor", [(.5, 2), (2, 3), (.2, 5), (-1, 2), (5, .2), (":)", 2)])
@pytest.mark.parametrize("n_pts", [10, 100])
def test_scale_augmentation(img_size: Tuple[int, ...],
                                  scaling_factor: Tuple[int, ...],
                                  n_pts: int,
                                  caplog):
    if caplog is not None:
        caplog.set_level(logging.CRITICAL)

    torch.manual_seed(img_size[-1]*n_pts)

    img = torch.zeros(img_size)
    msize = min(img_size[-2:])



    pts = torch.randint(msize//3, msize-msize//3, (n_pts, 2)).repeat(img_size[0], 1, 1)
    for b in range(img_size[0]):
        img[b] = torch.from_numpy(_generate_img_from_points(pts[b].numpy(), img_size[-2:], sigma=1))
    if any(not isinstance(sf, Number) or sf <= 0 for sf in scaling_factor) or scaling_factor[0] > scaling_factor[1]:
        with pytest.raises(ValueError):
            aug = IsotropicScale(order=1, scaling_factor=scaling_factor)
    else:
        aug = IsotropicScale(order=1, scaling_factor=scaling_factor)
        img_aug, pts_aug = aug(img, pts)
        img_from_aug_pts = torch.zeros(img_size)
        for b in range(img_size[0]):
            img_from_aug_pts[b] = torch.from_numpy(_generate_img_from_points(pts_aug[b].round().numpy(), img_size[-2:], sigma=1))
        mse = ((img_aug - img_from_aug_pts)**2).mean()
        if __name__ == "__main__":
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 4, figsize=(16, 4))
            ax[0].imshow(img[0], cmap="magma")
            ax[0].title.set_text("Original")
            ax[1].imshow(img_aug[0], cmap="magma")
            ax[1].title.set_text("Augmented")
            ax[2].imshow(img_from_aug_pts[0], cmap="magma")
            ax[2].title.set_text("From Augmented Points")
            ax[3].imshow((img_aug[0]-img_from_aug_pts[0])**2, cmap="magma")
            ax[3].title.set_text(f"Squared Difference (MSE: {mse:.5f})")
            fig.show()
        assert mse < MSE_TOLERANCE, "Image augmentation is not correct."

if __name__ == "__main__":
    test_scale_augmentation((1, 100, 100), (2, 3), 100, None)