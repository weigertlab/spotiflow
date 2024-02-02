from typing import Optional, Tuple
import itertools
import torch

from .base import BaseAugmentation
from .utils import _flatten_axis

def _subgroup_flips(ndim: int, axis: Optional[Tuple[int, ...]]=None) -> Tuple[Tuple[bool, ...], ...]:
    """Adapted from https://github.com/stardist/augmend/blob/main/augmend/transforms/affine.py not to depend on numpy
    iterate over the product subgroup (False,True) of given axis
    """
    axis = _flatten_axis(ndim, axis)
    res = [False for _ in range(ndim)]
    for prod in itertools.product((False, True), repeat=len(axis)):
        for a, p in zip(axis, prod):
            res[a] = p
        yield tuple(res)

def _fliprot_pts(pts: torch.Tensor, dims_to_flip: Tuple[int, ...], shape: Tuple[int, int], ndims: int) -> torch.Tensor:
    """Flip and rotate points accordingly to the flipping dimensions.

    Args:
        pts (torch.Tensor): points to be flipped and rotated.
        dims_to_flip (Tuple[int]): indices of the dimensions to be flipped.
        shape (Tuple[int, int]): shape of the image.

    Returns:
        torch.Tensor: flipped and rotated points.
    """
    y, x = shape
    pts_fr = pts.clone()
    for dim in dims_to_flip:
        if dim == ndims-2:
            pts_fr[..., 0] = y - 1 - pts_fr[..., 0]
        elif dim == ndims-1:
            pts_fr[..., 1] = x - 1 - pts_fr[..., 1]
    return pts_fr

class FlipRot90(BaseAugmentation):
    def __init__(self, probability: float=1.0) -> None:
        """Augmentation class for FlipRot90 augmentation.

        Args:
            probability (float): probability of applying the augmentation. Must be in [0, 1].

        """
        super().__init__(probability)

    def apply(self, img: torch.Tensor, pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies FlipRot90 augmentation to the given image and points.
        """
        # Randomly choose the spatial axis/axes to flip
        combs = tuple(_subgroup_flips(img.ndim, axis=(-2, -1)))
        idx = torch.randint(len(combs), (1,)).item()
        dims_to_flip = tuple(i for i, c in enumerate(combs[idx]) if c)

        # Return original image and points if no axis is flipped
        if len(dims_to_flip) == 0:
            return img, pts
        # Flip image and points

        img_fr = torch.flip(img, dims_to_flip)
        pts_fr = _fliprot_pts(pts, dims_to_flip, img.shape[-2:], ndims=img.ndim)
        return img_fr, pts_fr


    def __repr__(self) -> str:
        return f"FlipRot90(probability={self.probability})"