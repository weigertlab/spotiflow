from numbers import Number
from torchvision.transforms import InterpolationMode
from typing import Literal, Optional, Tuple
import torch
import torchvision.transforms.functional as tvf

from .base import BaseAugmentation
from .utils import _filter_points_idx

def _affine_scaling_matrix(scaling_factor: float, center:tuple[float]) -> torch.Tensor:
    """Generate a centered scale matrix transformation acting on 2D euclidean coordinates

    Args:
        scaling_factor (float): scaling factor

    Returns:
        torch.Tensor: scale transformation matrix
    """
    cy, cx = center
    translation_mat = torch.FloatTensor([
        [1., 0., -cy],
        [0., 1., -cx],
        [0., 0., 1.]
    ])
    translation_mat_inv = torch.FloatTensor([
        [1., 0., cy],
        [0., 1., cx],
        [0., 0., 1.]
    ])
    scaling_mat = torch.FloatTensor([
        [scaling_factor, 0., 0.],
        [0., scaling_factor, 0.],
        [0., 0., 1.]
    ])
    affine_mat = translation_mat_inv@scaling_mat@translation_mat
    return affine_mat


class IsotropicScale(BaseAugmentation):
    def __init__(self, order: Literal[0, 1]=1,
                 scaling_factor: Optional[Tuple[Number, Number]] = (0.5,2.0), probability: float=1.0) -> None:
        """Augmentation class for random isotropic scaling

        Args:
            probability (float): probability of applying the augmentation. Must be in [0, 1].
            order (Literal[0, 1]): order of interpolation. Use 0 for nearest neighbor, 1 for bilinear.
            scaling_factor (Optional[Tuple[Number, Number]], optional): scaling factor range to be sampled. If None, scaling factor is randomly sampled from [0.5, 2.0]. Defaults to None.
        """
        super().__init__(probability)
        self._order = int(order)
        if self._order == 0:
            self._interp_mode = InterpolationMode.NEAREST
        elif self._order == 1:
            self._interp_mode = InterpolationMode.BILINEAR
        else:
            raise ValueError("Order must be 0 or 1.")
        
        if all(isinstance(x, Number) for x in scaling_factor) and len(scaling_factor) == 2:
            scaling_factor = scaling_factor
        else:
            raise ValueError("Scaling factor must be a 2-length tuple of numbers.")
        self._scaling_factor = scaling_factor
        if any(sf <= 0 for sf in self._scaling_factor) or self._scaling_factor[0] > self._scaling_factor[1]:
            raise ValueError("Scaling factor must be in (0, +inf) and the first element must be smaller than the second one.")


    def apply(self, img: torch.Tensor, pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the augmentation to the input image and points

        Args:
            img (torch.Tensor): input image tensor of shape (C, H, W)
            pts (torch.Tensor): input points tensor of shape (N, 2) or (N, 3)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: transformed image tensor of shape (C, H, W) and transformed points tensor of shape (N, 2) or (N, 3)
        """
        # Sample scaling factor
        sampled_scaling_factor = self._sample_scaling_factor()
        img_scaled = tvf.affine(img,
                                angle=0,
                                translate=(0,0),
                                scale=sampled_scaling_factor,
                                shear=0,
                                interpolation=self._interp_mode)

        # Generate affine transformation matrix
        y, x = img.shape[-2:]
        center_y, center_x = (y-1)/2, (x-1)/2
        affine_mat = _affine_scaling_matrix(sampled_scaling_factor, (center_y, center_x)).to(img.device)

        # Scale points
        if pts.shape[2] == 2: # no class labels
            should_add_cls_label = False
            affine_coords = torch.cat([pts.float(), torch.ones((*pts.shape[:-1],1), device=img.device)], axis=-1) # Euclidean -> homogeneous coordinates
        else:
            should_add_cls_label = True
            affine_coords = torch.cat([pts[..., :-1].float(), torch.ones((*pts.shape[:-1],1), device=img.device)], axis=-1) # Euclidean -> homogeneous coordinates
        pts_scaled = (affine_coords@affine_mat.T)
        pts_scaled = pts_scaled[..., :-1] # Homogeneous -> Euclidean coordinates
        if should_add_cls_label:
            pts_scaled = torch.cat([pts_scaled, pts[..., -1:]], axis=-1)
        
        
        idxs_in = _filter_points_idx(pts_scaled, img_scaled.shape[-2:])
        return img_scaled, pts_scaled[idxs_in].view(*pts.shape[:-2], -1, pts.shape[-1])

    def _sample_scaling_factor(self) -> float:
        return torch.FloatTensor(1).uniform_(*self._scaling_factor).item()
    
    def __repr__(self) -> str:
        return f"IsotropicScale(scaling_factor={self._scaling_factor}, probability={self.probability})"

class AnisotropicScale(BaseAugmentation):
    # TODO
    def __init__(self, probability: float):
        raise NotImplementedError("AnisotropicScaleAugmentation is not implemented yet")
    