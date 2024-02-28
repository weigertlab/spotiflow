from typing import Tuple

import torch
import torchvision.transforms.functional as tvf

from .base import BaseAugmentation
from .utils import _filter_points_idx


class Crop(BaseAugmentation):
    def __init__(self, size: Tuple[int, int], probability: float=1.0, point_priority: float=0):
        """Augmentation class for random crops

        Args:
            probability (float): probability of applying the augmentation. Must be in [0, 1].
            size (Tuple[int, int]): size of the crop in (y, x) format
            priority (float): prioritizes crops centered around keypoints. Must be in [0, 1]. 

        """
        super().__init__(probability)
        assert len(size) == 2 and all([isinstance(s, int) for s in size]), "Size must be a 2-length tuple of integers"
        self._size = size
        self._point_priority = point_priority
    
    @property
    def size(self) -> Tuple[int, int]:
        return self._size
    
    @property
    def point_priority(self) -> float:
        return self._point_priority

    def apply(self, img: torch.Tensor, pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate random top-left anchor
        y, x = img.shape[-2:]
        assert y >= self.size[0] and x >= self.size[1], "Image is smaller than crop size"
        cy, cx = self._generate_tl_anchor(y, x, pts)
        # Crop image
        img_c = tvf.crop(img, top=cy, left=cx, height=self.size[0], width=self.size[1])

        # Crop points
        pts_c = pts - torch.FloatTensor([cy, cx])
        idxs_in = _filter_points_idx(pts_c, self.size)
        return img_c, pts_c[idxs_in].view(*pts.shape[:-2], -1, pts.shape[-1])
    
    def _generate_tl_anchor(self, y: int, x: int, pts: torch.Tensor) -> Tuple[int, int]:
        prob = torch.FloatTensor(1).uniform_(0, 1).item()
        
        if prob>self.point_priority:
            # Randomly generate top-left anchor
            cy, cx = torch.FloatTensor(1).uniform_(0, y-self.size[0]).item(), torch.FloatTensor(1).uniform_(0, x-self._size[1]).item()
            return int(cy), int(cx)
        else:
            width = self.size[0]//4, self.size[1]//4
            # Remove points that are not anchor candidates
            valid_pt_coords = pts[(pts[..., 0] >= self.size[0]//2-width[0]) & (pts[..., 0] < y-self.size[0]//2+width[0]) & (pts[..., 1] >= self.size[1]//2-width[1]) & (pts[..., 1] < x-self.size[1]//2+width[1])]
            if valid_pt_coords.shape[0] == 0:
                # sample randomly if no points are valid
                cy, cx = torch.FloatTensor(1).uniform_(0, y-self.size[0]).item(), torch.FloatTensor(1).uniform_(0, x-self._size[1]).item()
            else:
                # select a point 
                center_idx = torch.randint(0, valid_pt_coords.shape[0], (1,)).item()
                cy, cx = valid_pt_coords[center_idx]
                cy = cy +  torch.randint(-width[0], width[0]+1, (1,))
                cx = cx +  torch.randint(-width[1], width[1]+1, (1,))
                cy -= self.size[0]//2
                cx -= self.size[1]//2
                cy = torch.clip(cy, 0, y-self.size[0]).item()
                cx = torch.clip(cx, 0, x-self.size[1]).item()
        return int(cy), int(cx)

    def __repr__(self) -> str:
        return f"Crop(size={self.size}, probability={self.probability})"