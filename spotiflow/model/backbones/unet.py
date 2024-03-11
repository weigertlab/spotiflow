import torch
import torch.nn as nn

from typing import Iterable, Literal, Tuple, Union



class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int] = (3, 3),
        activation: nn.Module = nn.ReLU,
        batch_norm: bool = False,
        padding: Union[str, int] = "same",
        padding_mode: str = "zeros",
        dropout: float = 0,
        bias: bool = True,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
                bias=bias if bias and not batch_norm else False,
            ),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            activation(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)

class ConvBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int] = (3, 3, 3),
        activation: nn.Module = nn.ReLU,
        batch_norm: bool = False,
        padding: Union[str, int] = "same",
        padding_mode: str = "zeros",
        dropout: float = 0,
        bias: bool = True,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
                bias=bias if bias and not batch_norm else False,
            ),
            nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity(),
            activation(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)

class UNetBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        initial_fmaps: int,
        downsample_factors: Iterable[Tuple[int, int]] = ((2, 2), (2, 2)),
        fmap_inc_factor: int = 2,
        kernel_sizes: Tuple[Tuple[int, int]] = ((3, 3), (3, 3), (3, 3), (3, 3)),
        activation: nn.Module = nn.ReLU,
        batch_norm: bool = False,
        padding: Union[str, int] = 1,
        padding_mode: str = "zeros",
        concat_mode: Literal['cat', 'add'] = "cat",
        dropout: float = 0,
        upsampling_mode: str = "nearest",
        use_3d_convs: bool = False,
    ):

        super().__init__()

        ConvBlockModule = ConvBlock if not use_3d_convs else ConvBlock3D
        MaxPoolModule = nn.MaxPool2d if not use_3d_convs else nn.MaxPool3d

        self.levels = len(downsample_factors) + 1
        self.n_convs_per_level = len(kernel_sizes)

        parsed_kernel_sizes = []

        for k in kernel_sizes:
            if use_3d_convs and len(k) == 2:
                parsed_kernel_sizes.append((k[0], *k))
            elif not use_3d_convs and len(k) == 2:
                parsed_kernel_sizes.append(k)
            elif use_3d_convs and len(k) == 3:
                parsed_kernel_sizes.append(k)
            else:
                raise ValueError(f"Kernel size {k} must be of length 2 or 3. If 2D, then it must be (h, w). If 3D, then it must be (d, h, w).")
        parsed_kernel_sizes = tuple(parsed_kernel_sizes)

        parsed_downsample_factors = []
        for d in downsample_factors:
            if use_3d_convs and len(d) == 2:
                parsed_downsample_factors.append((d[0], *d))
            elif not use_3d_convs and len(d) == 2:
                parsed_downsample_factors.append(d)
            elif use_3d_convs and len(d) == 3:
                parsed_downsample_factors.append(d)
            else:
                raise ValueError(f"Downsample factor {d} must be of length 2 or 3. If 2D, then it must be (h, w). If 3D, then it must be (d, h, w).")

        self.concat_mode = concat_mode

        # Down
        self.down_blocks = nn.ModuleList()

        self.max_pools = nn.ModuleList(
            MaxPoolModule(kernel_size=parsed_downsample_factors[l])
            for l in range(self.levels - 1)
        )

        self.activation = activation

        for l in range(self.levels - 1):
            curr_lv = []
            for n in range(self.n_convs_per_level):
                curr_lv += [
                    ConvBlockModule(
                        in_channels=in_channels
                        if l == 0 and n == 0
                        else int(initial_fmaps * fmap_inc_factor ** (l - 1))
                        if n == 0
                        else int(initial_fmaps * fmap_inc_factor**l),
                        out_channels=int(initial_fmaps * fmap_inc_factor**l),
                        kernel_size=parsed_kernel_sizes[n],
                        activation=self.activation,
                        batch_norm=batch_norm,
                        padding=padding,
                        padding_mode=padding_mode,
                        dropout=dropout,
                    )
                ]
            self.down_blocks.append(nn.Sequential(*curr_lv))

        # Middle
        self.middle = nn.Sequential(
            *[
                ConvBlockModule(
                    in_channels=int(
                        initial_fmaps * fmap_inc_factor ** (self.levels - 2)
                        if n == 0
                        else initial_fmaps * fmap_inc_factor ** (self.levels - 1)
                    ),
                    out_channels=int(
                        initial_fmaps * fmap_inc_factor ** (self.levels - 1)
                    ),
                    kernel_size=parsed_kernel_sizes[self.n_convs_per_level - 1],
                    activation=self.activation,
                    batch_norm=batch_norm,
                    padding=padding,
                    padding_mode=padding_mode,
                    dropout=dropout,
                )
                for n in range(self.n_convs_per_level - 1)
            ]
            + [
                ConvBlockModule(
                    in_channels=int(
                        initial_fmaps * fmap_inc_factor ** (self.levels - 1)
                    ),
                    out_channels=int(
                        initial_fmaps * fmap_inc_factor ** max(self.levels - 2, 0)
                    ),
                    kernel_size=parsed_kernel_sizes[self.n_convs_per_level - 1],
                    activation=self.activation,
                    batch_norm=batch_norm,
                    padding=padding,
                    padding_mode=padding_mode,
                    dropout=dropout,
                )
            ]
        )

        parsed_downsample_factors = tuple((tuple(d) for d in parsed_downsample_factors))

        self.upsamples = nn.ModuleList(
            [
                nn.Upsample(scale_factor=parsed_downsample_factors[l], mode=upsampling_mode)
                for l in reversed(range(self.levels - 1))
            ]
        )

        self.up_blocks = []

        # Up with skip layers
        for l in reversed(range(self.levels - 1)):
            curr_lv = []
            for n in range(self.n_convs_per_level - 1):
                if n==0:
                    if self.concat_mode=='cat':
                        in_channels = 2* int(initial_fmaps * fmap_inc_factor**l)
                    elif self.concat_mode=='add':
                        in_channels = int(initial_fmaps * fmap_inc_factor **l)
                    else: 
                        raise ValueError(f'concat_mode {self.concat_mode} must be either "cat" or "add"')
                else:
                    in_channels = int(initial_fmaps * fmap_inc_factor**l)
                curr_lv += [
                    ConvBlockModule(
                        in_channels=in_channels,
                        out_channels=int(initial_fmaps * fmap_inc_factor**l),
                        kernel_size=parsed_kernel_sizes[n],
                        activation=self.activation,
                        batch_norm=batch_norm,
                        padding=padding,
                        padding_mode=padding_mode,
                        dropout=dropout,
                    )
                ]
            curr_lv += [
                ConvBlockModule(
                    in_channels=int(initial_fmaps * fmap_inc_factor**l),
                    out_channels=int(initial_fmaps * fmap_inc_factor ** max(0, l - 1)),
                    kernel_size=parsed_kernel_sizes[self.n_convs_per_level - 1],
                    activation=self.activation,
                    batch_norm=batch_norm,
                    padding=padding,
                    padding_mode=padding_mode,
                    dropout=dropout,
                )
            ]
            self.up_blocks += [nn.Sequential(*curr_lv)]

        self.up_blocks = nn.ModuleList(self.up_blocks[::-1])

        self.out_channels_list = [
            int(initial_fmaps * fmap_inc_factor ** max(0, l - 1))
            for l in range(self.levels - 1)
        ] + [int(initial_fmaps * fmap_inc_factor ** max(self.levels - 2, 0))]

        def init_kaiming(m):
            if self.activation is nn.ReLU:
                nonlinearity = "relu"
            elif self.activation is nn.LeakyReLU:
                nonlinearity = "leaky_relu"
            else:
                raise ValueError(
                    f"Kaiming init not applicable for activation {self.activation}."
                )
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if activation is nn.ReLU or activation is nn.LeakyReLU:
            self.apply(init_kaiming)

    def forward(self, x: torch.Tensor):
        skip_layers = []
        # Down...
        for l in range(self.levels - 1):
            x = self.down_blocks[l](x)
            skip_layers.append(x)
            x = self.max_pools[l](x)

        # Middle
        x = self.middle(x)
        out = [x]

        # Up in reverse
        for l in reversed(range(self.levels - 1)):
            x = self.upsamples[l](x)
            if self.concat_mode=='cat':
                x = torch.cat([x, skip_layers[l]], dim=1)
            elif self.concat_mode=='add':
                x = x + skip_layers[l]
            else:
                raise ValueError(self.concat_mode)
            
            x = self.up_blocks[l](x)
            out += [x]

        return out[::-1]  # Reverse the list to get finer to coarser outputs


if __name__ == "__main__":
    import sys

    t = torch.randn(4, 1, 8, 64, 64)
    # t = torch.randn(4, 1, 64, 64)
    model = UNetBackbone(
        in_channels=1,
        initial_fmaps=16,
        padding="same",
        downsample_factors=((2, 2), (2, 2), (2, 2)),
        kernel_sizes=((3, 3), (3, 3)),
        use_3d_convs=True,
    )
    print(f"Resolution levels: {model.levels}")
    print(f"Number of UNet parameters: {sum(p.numel() for p in model.parameters())}")
    pred = model(t)
    print(model.out_channels_list)
    for i, p in enumerate(pred):
        print(f"Level {i}: {p.shape}")
    sys.exit(0)
