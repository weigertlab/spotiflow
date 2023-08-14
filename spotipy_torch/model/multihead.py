import torch
import torch.nn as nn
from typing import List, Tuple, Union

from .backbones.unet import ConvBlock

class MultiHeadProcessor(nn.Module):
    def __init__(self, in_channels_list: List[int],
                       out_channels: int,
                       kernel_sizes: Tuple[Tuple[int, int]],
                       initial_fmaps: int,
                       fmap_inc_factor: int=2,
                       activation: nn.Module=nn.LeakyReLU,
                       padding: Union[str, int]="same",
                       padding_mode: str="zeros",
                       dropout: int=0,
                 ):
        super().__init__()
        self.n_heads = len(in_channels_list)
        self.n_convs_per_head = len(kernel_sizes)
        self.activation = activation
        self.heads = nn.ModuleList()
        self.last_convs = nn.ModuleList() # Need to be separated to avoid kaiming init on blocks with non-relu activations
        for h in range(self.n_heads):
            curr_head = []
            for n in range(self.n_convs_per_head):
                curr_head += [
                    ConvBlock(in_channels=in_channels_list[h] if n == 0 else initial_fmaps * fmap_inc_factor ** h,
                              out_channels=initial_fmaps * fmap_inc_factor ** h,
                              kernel_size=kernel_sizes[n],
                              activation=self.activation,
                              padding=padding,
                              padding_mode=padding_mode,
                              dropout=dropout)
                ]
            self.heads.append(nn.Sequential(*curr_head))
            self.last_convs.append(ConvBlock(in_channels=initial_fmaps * fmap_inc_factor ** h,
                                             out_channels=out_channels,
                                             kernel_size=1,
                                             activation=nn.Identity))

        def init_kaiming(m):
            if self.activation is nn.ReLU:
                nonlinearity = "relu"
            elif self.activation is nn.LeakyReLU:
                nonlinearity = "leaky_relu"
            else:
                raise ValueError(
                    f"Kaiming init not applicable for activation {self.activation}."
                )
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
                nn.init.zeros_(m.bias)

        if activation is nn.ReLU or activation is nn.LeakyReLU:
            self.heads.apply(init_kaiming)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(x) == self.n_heads, f"Expected {self.n_heads} inputs, got {len(x)}"
        out = [None]*self.n_heads
        for h in range(self.n_heads):
            out[h] = self.heads[h](x[h])
            out[h] = self.last_convs[h](out[h])
        return out
