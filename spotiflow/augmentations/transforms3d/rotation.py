from numbers import Number
from torchvision.transforms import InterpolationMode
from typing import Literal, Optional, Tuple
import math
import torch
import torchvision.transforms.functional as tvf

from ..transforms.base import BaseAugmentation
from ..transforms.utils import _filter_points_idx

def _affine_rotation_matrix(phi:float, center:tuple[float]) -> torch.Tensor:
    cz, cy, cx = center
    translation_mat = torch.FloatTensor([
        [1., 0., 0., -cz],
        [0., 1., 0., -cy],
        [0., 0., 1., -cx],
        [0., 0., 0., 1.]
    ])
    translation_mat_inv = torch.FloatTensor([
        [1., 0., 0., cz],
        [0., 1., 0., cy],
        [0., 0., 1., cx],
        [0., 0., 0., 1.]
    ])

    si, co = math.sin(phi), math.cos(phi)

    rot_matrix_yx = torch.FloatTensor([
        [1., 0., 0., 0.],
        [0., co, si, 0.],
        [0., -si, co, 0.],
        [0., 0., 0., 1.]
    ])
    affine_mat = translation_mat_inv@rot_matrix_yx@translation_mat
    return affine_mat


class RotationYX3D(BaseAugmentation):
    def __init__(self, order: Literal[0, 1]=1, angle: Optional[Number] = (-180,180), probability: float=1.0) -> None:
        """Augmentation class for random rotations (YX plane).

        Assuming the last three dimensions of the image tensor are spatial dimensions (z, y, x).

        Args:
            probability (float): probability of applying the augmentation. Must be in [0, 1].
            order (Literal[0, 1]): order of interpolation. Use 0 for nearest neighbor, 1 for bilinear.
            angle (Optional[float]): +/- rotation angle (in degrees). If None, angle is randomly sampled from [-180,180]. Defaults to None.

        """
        super().__init__(probability)
        self._order = int(order)
        if self._order == 0:
            self._interp_mode = InterpolationMode.NEAREST
        elif self._order == 1:
            self._interp_mode = InterpolationMode.BILINEAR
        else:
            raise ValueError("Order must be 0 or 1.")
        if angle is None:
            angle = (-180, 180)
        elif isinstance(angle, Number):
            angle = (-angle, angle)
        elif isinstance(angle, tuple) and len(angle) == 2 and all(isinstance(x, Number) for x in angle):
            angle = angle
        else:
            raise ValueError("angle must be either a number or a tuple of two numbers")
        self._angle = angle
        if all(phi == 0 for phi in angle):
            raise ValueError("Angle range cannot be (0, 0).")

    def apply(self, img: torch.Tensor, pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate random rotation angle
        phi_deg = self._sample_angle()
        phi_rad = phi_deg * math.pi / 180.

        # Rotate image
        assert img.ndim < 5 or img.shape[0] == 1, "Leading axis dimensionality should be 1 for volumetric data"

        if should_unsqueeze := (img.ndim == 5):
            img = img.squeeze(0)
        img_r = tvf.rotate(img, phi_deg, interpolation=self._interp_mode)

        if should_unsqueeze:
            img_r = img_r.unsqueeze(0)

        # Generate affine transformation matrix
        y, x = img.shape[-2:]
        center_z, center_y, center_x = 0, (y-1)/2, (x-1)/2

        affine_mat = _affine_rotation_matrix(-phi_rad, (center_z, center_y, center_x)).to(img.device)
        # Rotate points
        affine_coords = torch.cat([pts.float(), torch.ones((*pts.shape[:-1],1), device=img.device)], axis=-1) # Euclidean -> homogeneous coordinates
        pts_r = (affine_coords@affine_mat.T)
        pts_r = pts_r[..., :-1] # Homogeneous -> Euclidean coordinates

        idxs_in = _filter_points_idx(pts_r, img_r.shape[-3:])

        return img_r, pts_r[idxs_in].view(*pts.shape[:-2], -1, pts.shape[-1])

    def _sample_angle(self):
        return torch.FloatTensor(1).uniform_(*self._angle).item()


    def __repr__(self) -> str:
        return f"Rotation(angle={self._angle}, probability={self.probability})"