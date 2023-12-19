from numbers import Number
from typing import Tuple
import torch

from .base import BaseAugmentation


class GaussianNoise(BaseAugmentation):
    def __init__(self, sigma: Tuple[float, float]=(0., 0.05), probability: float=1.0) -> None:
        """Augmentation class for shifting and scaling the intensity of the image.
        
        I = I * scale + shift

        Args:
            probability (float): probability of applying the augmentation. Must be in [0, 1].
            scale (Tuple[int, int]): range of the scaling factor to apply to the image. 
            shift (Tuple[int, int]): range of the shift to apply to the image. 
        """
        super().__init__(probability)
        assert len(sigma) == 2 and all([isinstance(s, Number) for s in sigma]), "Sigma must be a 2-length tuple of floating point numbers."
        self._sigma = sigma

    def apply(self, img: torch.Tensor, pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies IntensityScaleShift augmentation to the given image and points.
        Args:
            img (torch.Tensor): image to be augmented.
            pts (torch.Tensor): points to be augmented.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: augmented image and points.
        """
        # Randomly choose the scaling factor and shift
        sampled_sigma = self._sample_sigma()
        noise = torch.randn_like(img, dtype=torch.float32, device=img.device) * sampled_sigma
        return img + noise, pts

    def _sample_sigma(self):
        return torch.empty(1, dtype=torch.float32).uniform_(*self._sigma)
    
    def __repr__(self) -> str:
        return f"GaussianNoise(sigma={self._sigma}, probability={self.probability})"

class SaltAndPepperNoise(BaseAugmentation):
    def __init__(self, prob_pepper: Tuple[float, float]=(0., 0.001), prob_salt: Tuple[float, float]=(0., 0.001), probability: float=1.0):
        """Augmentation class for addition of salt and pepper noise to the image.
        
        Each pixel is randomly set to the minimum (pepper) or maximum (salt) value of the image with a given probability.

        Args:
            probability (float): probability of applying the augmentation. Must be in [0, 1].
            prob_pepper (Tuple[float, float]): range of the potential probability of adding pepper noise to the image.
            prob_salt (Tuple[float, float]): range of the potential probability of adding salt noise to the image.
        """
        super().__init__(probability)

        assert len(prob_pepper) == 2 and \
               all([isinstance(s, Number) for s in prob_pepper]) and \
               all([s >= 0 and s<= 1 for s in prob_pepper]), "prob_pepper must be a 2-length tuple of floating point numbers between zero and one."

        assert len(prob_salt) == 2 and \
               all([isinstance(s, Number) for s in prob_salt]) and \
               all([s >= 0 and s<= 1 for s in prob_salt]), "prob_salt must be a 2-length tuple of floating point numbers between zero and one."
        
        assert prob_pepper[0] <= prob_pepper[1], "First element of prob_pepper must be smaller or equal to the second element."
        assert prob_salt[0] <= prob_salt[1], "First element of prob_salt must be smaller or equal to the second element."
        
        self._prob_salt = prob_salt
        self._prob_pepper = prob_pepper
    
    def apply(self, img: torch.Tensor, pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sampled_prob_pepper = self._sample_prob_pepper()
        sampled_prob_salt = self._sample_prob_salt()
        
        pepper_mask = torch.empty_like(img, dtype=torch.float32).uniform_(0, 1)
        salt_mask = torch.empty_like(img, dtype=torch.float32).uniform_(0, 1)

        pepper_value = img.min()
        salt_value = img.max()

        img_sp = img.clone()
        img_sp[pepper_mask < sampled_prob_pepper] = pepper_value
        img_sp[salt_mask < sampled_prob_salt] = salt_value
        return img_sp, pts

    def _sample_prob_pepper(self):
        return torch.empty(1, dtype=torch.float32).uniform_(*self._prob_pepper)

    def _sample_prob_salt(self):
        return torch.empty(1, dtype=torch.float32).uniform_(*self._prob_salt)


    def __repr__(self) -> str:
        return f"SaltAndPepperNoise(prob_pepper={self._prob_pepper}, prob_salt={self._prob_salt}, probability={self.probability})"