from typing import Tuple
import torch

from .base import BaseAugmentation


class IntensityScaleShift(BaseAugmentation):
    def __init__(self, scale: Tuple[float, float]=(.8, 1.2),
                 shift: Tuple[float, float]=(-.1, .1), probability: float=1.0) -> None:
        """Augmentation class for shifting and scaling the intensity of the image.
        
        I = I * scale + shift

        Args:
            probability (float): probability of applying the augmentation. Must be in [0, 1].
            scale (Tuple[int, int]): range of the scaling factor to apply to the image. 
            shift (Tuple[int, int]): range of the shift to apply to the image. 
        """
        super().__init__(probability)
        assert len(scale) == 2 and all([isinstance(s, float) for s in scale]), "Scale must be a 2-length tuple of floating point numbers."
        assert len(shift) == 2 and all([isinstance(s, float) for s in shift]), "Shift must be a 2-length tuple of floating point numbers."
        assert scale[0] <= scale[1], "First element of scale must be smaller or equal to the second element."
        assert shift[0] <= shift[1], "First element of shift must be smaller or equal to the second element." 
        self._scale = scale
        self._shift = shift


    def apply(self, img: torch.Tensor, pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies IntensityScaleShift augmentation to the given image and points.
        Args:
            img (torch.Tensor): image to be augmented.
            pts (torch.Tensor): points to be augmented.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: augmented image and points.
        """
        # Randomly choose the scaling factor and shift
        sampled_scale = self._sample_scale(img.device)
        sampled_shift = self._sample_shift(img.device)
        return sampled_scale*img + sampled_shift, pts
    
    def _sample_scale(self, device):
        return torch.empty(1, dtype=torch.float32, device=device).uniform_(*self._scale)
    
    def _sample_shift(self, device):
        return torch.empty(1, dtype=torch.float32, device=device).uniform_(*self._shift)

    def __repr__(self) -> str:
        return f"IntensityScaleShift(scale={self._scale}, shift={self._shift}, probability={self.probability})"