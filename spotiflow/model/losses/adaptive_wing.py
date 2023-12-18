import torch
import torch.nn as nn
import scipy.ndimage
import numpy as np

class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing loss (Wang et al., ICCV 2019)

    Args:
        theta (float): Threshold between linear and non linear loss.
        alpha (float): Used to adapt loss shape to input shape and make loss smooth at 0 (background).
        It needs to be slightly above 2 to maintain ideal properties.
        omega (float): Multiplicating factor for non linear part of the loss.
        epsilon (float): factor to avoid gradient explosion. It must not be too small
        reduction (str): function reduction applied to loss
    """

    def __init__(self, theta=0.5, alpha=2.1, omega=14, epsilon=1, reduction="none"):
        super().__init__()
        self.theta = theta
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self._reduction = reduction

    def forward(self, input, target):
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon,
                                             self.alpha - target))) * \
            (self.alpha - target) * torch.pow(self.theta / self.epsilon,
                                              self.alpha - target - 1) * (1 / self.epsilon)
        C = (self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - target)))

        abs_diff = torch.abs(input - target)
        idx_small = abs_diff < self.theta
        idx_large = abs_diff >= self.theta

        loss = torch.zeros_like(input)
       
        loss[idx_small] = self.omega*torch.log(1+torch.pow(abs_diff[idx_small]/self.epsilon, self.alpha-target[idx_small]))
        loss[idx_large] = A[idx_large]*abs_diff[idx_large] - C[idx_large]
        if self._reduction == "none":
            return loss
        elif self._reduction == "sum":
            return loss.sum()
        elif self._reduction == "mean":
            return loss.mean()
        else:
            raise ValueError(f"Invalid reduction {self._reduction}")


if __name__ == "__main__":
    inp = torch.clip(torch.randn(3, 1, 512, 512), 0, 1)
    target = torch.clip(torch.randn(3, 1, 512, 512), 0, 1)
    loss = AdaptiveWingLoss(reduction="sum")
    print(loss(inp, target))
