from typing import List, Tuple

import torch

from ..transforms.base import BaseAugmentation

class Pipeline(object):
    def __init__(self, *augs) -> None:
        super().__init__()
        self._augmentations = []
        for aug in augs:
            self.add(aug)
        

    @property
    def augmentations(self) -> List[BaseAugmentation]:
        """Ordered augmentations in the pipeline.

        Returns:
            List[BaseAugmentation]: Ordered augmentations in the pipeline
        """
        return self._augmentations

    def add(self, augmentation: BaseAugmentation):
        """Add a new augmentation to the pipeline.

        Args:
            augmentation (BaseAugmentation): augmentation object to be added
        """
        if not isinstance(augmentation, BaseAugmentation):
            raise TypeError("Only BaseAugmentation instances can be added")
        self._augmentations.append(augmentation)

    def __call__(self, img: torch.Tensor, pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations sequentially to an image and the corresponding points.

        Args:
            img (torch.Tensor): (N+1)D tensor of shape (C, ..., H, W)
            pts (torch.Tensor): 2D tensor of shape (N, D)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: augmented batch of images and points
        """
        for aug in self.augmentations:
            img, pts = aug(img, pts)
        return img, pts


    def __repr__(self) -> str:
        aug_list = '\n- '.join(str(aug) for aug in self.augmentations)
        return f"Pipeline\n- {aug_list}"