from torchvision.transforms import InterpolationMode
from typing import Literal, Tuple
import torch
import torchvision.transforms.functional as tvf
from numbers import Number
from .base import BaseAugmentation
from .utils import _filter_points_idx

class Translation(BaseAugmentation):
    def __init__(self, order: Literal[0, 1]=1, shift: Tuple[int, int]=(-5,5), probability: float=1.0) -> None:
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
        
        if isinstance(shift, Number):
            shift = (-shift, shift)
        elif all(isinstance(x, int) for x in shift) and len(shift) == 2:
            shift = shift
        else:
            raise ValueError("Shift must be either a single integer or a 2-length tuple of integers.")
        self._shift = shift
        if all(x == 0 for x in shift):
            raise ValueError("Shift range cannot be (0, 0).")


    def apply(self, img: torch.Tensor, pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the augmentation to the input image and points

        Args:
            img (torch.Tensor): input image tensor of shape (C, H, W)
            pts (torch.Tensor): input points tensor of shape (N, 2) or (N, 3)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: transformed image tensor of shape (C, H, W) and transformed points tensor of shape (N, 2) or (N, 3)
        """
        # Sample shift
        sampled_translation = self._sample_shift(img.device)
        tr_x, tr_y = sampled_translation
        # Skip if no translation
        if tr_y == 0 and tr_x == 0:
            return img, pts
        
        # Apply translation to image and points
        img_translated = tvf.affine(img,
                                angle=0,
                                translate=(tr_y, tr_x), # First axis arg is horizontal shift, second is vertical shift
                                scale=1,
                                shear=0,
                                interpolation=self._interp_mode)
        pts_translated = pts + sampled_translation
        idxs_in = _filter_points_idx(pts_translated, img_translated.shape[-2:])
        pts_translated = pts_translated[idxs_in].view(*pts.shape[:-2], -1, pts.shape[-1])
        return img_translated, pts_translated
    

    def _sample_shift(self, device: torch.device) -> torch.Tensor:
        """Sample a random shift from the shift range

        Returns:
            torch.Tensor: random shift
        """
        return torch.randint(*self._shift, size=(2,), device=device)


    def __repr__(self) -> str:
        return f"Translation(shift={self._shift}, probability={self.probability})"