from collections import OrderedDict
from typing import List, Optional
import torch.nn as nn
import torch.nn.functional as F

class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network (Lin et al., CVPR '17) custom implementation allowing for different 
       interpolation modes as well as extra control
    """
    def __init__(self, in_channels_list: List[int],
                       out_channels: int,
                       extra_blocks: Optional[nn.Module]=None,
                       bias: bool=False,
                       interpolation_mode: str="bilinear",
                       align_corners: bool=True,
                       extra_modules: Optional[nn.ModuleList]=None) -> None:
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.extra_blocks = extra_blocks
        self.premergers = nn.ModuleList([
            nn.Conv2d(in_channels=c, out_channels=out_channels, kernel_size=1, padding=0, bias=bias)
        for c in in_channels_list])
        self.smoothers = nn.ModuleList([
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=bias)
        for _ in in_channels_list])
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners if self.interpolation_mode in ["linear", "bilinear", "bicubic", "trilinear"] else None
        self.extra_modules = extra_modules
        if self.extra_modules is not None:
            assert len(self.extra_modules) == len(self.in_channels_list), "One extra module should be provided per feature map"
    
    def forward(self, feature_maps) -> OrderedDict:
        """Assumes highest resolution is first, lowest is last in the input object.
        Args:
            obj (OrderedDict): features maps computed from the backbone. Assumes highest resolution is first, lowest is last in the input object.

        Returns:
            OrderedDict: FPN-processed feature maps.
        """
        fpn_outputs = [None]*len(feature_maps)
        fpn_outputs[-1] = self.premergers[-1](feature_maps[-1])
        for idx in reversed(range(0, len(fpn_outputs)-1)): # lowest to highest
            # Upsample previous FPN output
            upsampled = F.interpolate(fpn_outputs[idx+1],
                                      size=feature_maps[idx].shape[-2:],
                                      mode=self.interpolation_mode,
                                      align_corners=self.align_corners)
            # Add a 1x1-conv processed version of the current feature map to the upsampled previous FPN level
            fpn_outputs[idx] = upsampled+self.premergers[idx](feature_maps[idx])
        
        # Run the final 3x3 convolution independently on the FPN outputs
        smoothed_fpn_output = [self.smoothers[i](val) for i, val in enumerate(fpn_outputs)]
        return smoothed_fpn_output
    