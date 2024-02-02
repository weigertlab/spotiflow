from numbers import Number

import logging
import pytest
import torch

from spotiflow.augmentations.transforms import FlipRot90
from spotiflow.augmentations.transforms.utils import _generate_img_from_points
from typing import Tuple

MSE_TOLERANCE = 1e-8


@pytest.mark.parametrize("img_size", [(4, 1, 224, 224), (5, 2, 512, 512), (1, 100, 100),
                                      (3, 327, 312), (8, 100, 101), (10, 242, 256)])
@pytest.mark.parametrize("n_pts", [10, 100])
def test_fliprot90_augmentation(img_size: Tuple[int, ...],
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
    
    aug = FlipRot90()
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
    assert mse < MSE_TOLERANCE, "FlipRot90 augmentation is not correct."

if __name__ == "__main__":
    test_fliprot90_augmentation((10, 242, 256), 100, None)