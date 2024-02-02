import torch
import torch.nn as nn
import torch.nn.functional as F

class BackgroundRemover(nn.Module):
    """Remove background of an input image I_in by substracting a low-pass filtered
    of the image (I_low) from it. That is, I_out = I_in-I_low. The convolving filter is
    a large radius Gaussian kernel. Note that this is disabled by default.


    Args:
        n_channels (int, optional): number of channels in the input image. Defaults to 1.
        radius (int, optional): Gaussian filter radius (in px.). Defaults to 51.
    """
    def __init__(self, n_channels: int=1, radius: int=51) -> None:
        super().__init__()
        assert n_channels == 1, "Only 1-channel images are currently supported"
        assert radius % 2 == 1, "Radius must be odd"
        self._half_radius = int(radius)//2
        h = torch.exp(-torch.linspace(-2,2,2*self._half_radius+1)**2).float()
        h /= torch.sum(h)

        self.register_buffer("wy", h.reshape((1,1,len(h),1)))
        self.register_buffer("wx", h.reshape((1,1,1,len(h))))

        self.wy.requires_grad = False
        self.wx.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x, pad=tuple(4*[self._half_radius]), mode="reflect")
        y = F.conv2d(y, weight=self.wy, stride=1, padding="valid")
        y = F.conv2d(y, weight=self.wx, stride=1, padding="valid")
        return x - y