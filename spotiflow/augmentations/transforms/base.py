import abc
from typing import Tuple
import torch

class BaseAugmentation(abc.ABC):
    def __init__(self, probability: float, **kwargs):
        assert 0 <= probability <= 1
        self._probability = probability

    @property
    def probability(self) -> float:
        """Probability of applying every augmentation at every call

        Returns:
            float: probability of applying every augmentation at every call
        """
        return self._probability

    @abc.abstractmethod
    def apply(self, img: torch.Tensor, pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ implementation of the augmentation"""
        pass

    def __call__(self, img: torch.Tensor, pts: torch.Tensor):
        if self._should_apply():
            return self.apply(img, pts)
        else: 
            return img, pts

    def _should_apply(self) -> bool:
        """Sample from a [0,1) uniform distribution and compare to the probability
           of applying the augmentation in order to decide whether to apply
           the augmentation or not. 

        Returns:
            bool: return whether the augmentation should be applied
        """
        return torch.rand(()) < self.probability
