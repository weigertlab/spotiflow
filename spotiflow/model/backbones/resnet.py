"""Adapted from https://github.com/bentaculum/backbones/blob/main/backbones/resnet.py"""
import logging
import numpy as np
from torch import nn

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """ResidualBlock.

    Args:

        in_channels (``int``):

            The number of input channels for the first convolution in the
            block.

        out_channels (``int``):

            The number of output channels for all convolutions in the
            block.

        kernel_sizes (``tuple`` of ``int``):

            The number of tuples determines the number of
            convolutional layers in each ConvBlock. If not given, each
            ConvBlock will consist of two 3x3 convolutions.

        downsample_factor (``int`` or tuple of ``int``):

            Use as stride in the first convolution.

        activation (``torch.nn.Module``):

            Which activation to use after a convolution.

        batch_norm (``bool``):

            If set to ``True``, apply 2d batch normalization after each
            convolution.

        group_norm (``int``):

            Number of disjunct groups for group normalization.
            If set to ``False`` group normalization is not applied.

        padding (``int``):

            Padding added to both sides of the input. Defaults to 0.

        padding_mode (``str``):

            `torch.nn.Conv2d` padding modes: `zeros`, `reflect`, `replicate` or
            `circular`.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=[(3, 3)],
        downsample_factor=2,
        activation=nn.LeakyReLU,
        batch_norm=True,
        group_norm=False,
        padding=1,
        padding_mode="zeros",
        dropout=0,
    ):

        super().__init__()

        conv_block = []
        rec_in = in_channels
        dims = len(kernel_sizes[0])
        conv = {
            2: nn.Conv2d,
            3: nn.Conv3d,
        }[dims]
        batchnorm = {
            2: nn.BatchNorm2d,
            3: nn.BatchNorm3d,
        }[dims]

        for i, k in enumerate(kernel_sizes):
            # If kernel size is 1 in a dimension, do not pad.
            dim_padding = tuple(0 if _k == 1 else padding for _k in k)

            conv_block.append(
                conv(
                    in_channels=rec_in,
                    out_channels=out_channels,
                    kernel_size=k,
                    padding=dim_padding,
                    padding_mode=padding_mode,
                    stride=downsample_factor if i == 0 else 1,
                )
            )
            if batch_norm:
                conv_block.append(batchnorm(out_channels))
            if isinstance(group_norm, int) and group_norm > 0:
                conv_block.append(
                    nn.GroupNorm(
                        num_groups=group_norm,
                        num_channels=out_channels,
                    )
                )
            if i < len(kernel_sizes) - 1:
                try:
                    conv_block.append(activation(inplace=True))
                except TypeError:
                    conv_block.append(activation())

            rec_in = out_channels
        self.conv_block = nn.Sequential(*conv_block)

        shortcut = [
            conv(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=downsample_factor,
            )
        ]
        if batch_norm:
            shortcut.append(batchnorm(out_channels))
        if isinstance(group_norm, int) and group_norm > 0:
            shortcut.append(
                nn.GroupNorm(
                    num_groups=group_norm,
                    num_channels=out_channels,
                )
            )
        self.shortcut = nn.Sequential(*shortcut)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        try:
            self.final_activation = activation(inplace=True)
        except TypeError:
            self.final_activation = activation()

    def forward(self, x):

        return self.dropout(self.final_activation(self.conv_block(x) + self.shortcut(x)))


class Resnet(nn.Module):
    """Configurable Resnet-like CNN, for 2d and 3d inputs.

    Input tensors are expected to be of shape ``(B, C, (Z), Y, X)``.

    Args:

        in_channels:

            The number of input channels.

        inital_fmaps:

            The number of feature maps in the first layer. This is also the
            number of output feature maps. Stored in the ``channels``
            dimension.

        downsample_factors:

            Tuple of tuple of ints to use to downsample the
            feature in each residual block.

        fmap_inc_factor:

            By how much to multiply the number of feature maps between
            blocks. If block 0 has ``k`` feature maps, layer ``l`` will
            have ``k*fmap_inc_factor**l``.

        kernel_sizes (optional):

            Tuple of of tuple of ints, or tuple of tuple of tuple of ints.
            The number of ints determines the number of convolutional layers in
            each residual block. If not given, each block will consist of two
            3x3 convolutions.

        activation (``torch.nn.Module``):

            Which activation to use after a convolution.

        batch_norm (optional):

            If set to ``True``, apply 2d batch normalization after each
            convolution in the ConvBlocks.

        group_norm (``int``):

            Number of disjunct groups for group normalization.
            If set to ``False`` group normalization is not applied.

        padding (``int``):

            Padding added to both sides of the convolutions. Defaults to 0.

        padding_mode (``str``):

            `torch.nn.Conv2/3d` padding modes: `zeros`, `reflect`, `replicate`
            or `circular`.
    """

    def __init__(
        self,
        in_channels,
        initial_fmaps,
        downsample_factors=((2, 2), (2, 2), (2, 2)),
        fmap_inc_factor=2,
        kernel_sizes=(((3, 3), (3, 3),),),
        activation=nn.LeakyReLU,
        batch_norm=True,
        group_norm=False,
        padding=1,
        padding_mode="zeros",
        dropout=0,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.initial_fmaps = initial_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factors = downsample_factors
        self.kernel_sizes = kernel_sizes
        self.activation = activation

        if group_norm and batch_norm:
            raise ValueError("Do not apply multiple normalization approaches.")
        self.batch_norm = batch_norm
        if group_norm > initial_fmaps:
            raise ValueError(f"{group_norm=} bigger {initial_fmaps=}.")
        self.group_norm = group_norm

        try:
            assert isinstance(kernel_sizes[0][0][0], int)
        except (TypeError, AssertionError):
            raise ValueError("kernel_sizes expected to be a 3-level nested"
                             "tuple (network layer, convs per block, spatial dims)")

        try:
            assert isinstance(downsample_factors[0][0], int)
        except (TypeError, AssertionError):
            raise ValueError("downsample factors expected to be a 2-level"
                             " nested tuple(conv block, spatial dims)")

        if not (len(kernel_sizes) == 1 or len(kernel_sizes) == len(downsample_factors)):
            raise ValueError("kernel_sizes and downsamle_factors dimensionalities do not correspond.")
        if len(kernel_sizes) == 1:
            kernel_sizes = kernel_sizes * len(downsample_factors)

        for k in kernel_sizes:
            if not np.all(np.logical_or((np.array(k) - 1) / 2 == padding, np.array(k) == 1)):
                raise NotImplementedError("Only `same` padding implemented.")

        self.padding = padding
        self.padding_mode = padding_mode

        self.levels = len(downsample_factors)
        self.out_channels_list = [int(initial_fmaps * fmap_inc_factor**level) for level in range(self.levels)] # To be used by the FPN

        # TODO parametrize input block
        self.input_block = nn.Identity()

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=in_channels
                    if level == 0
                    else int(initial_fmaps * fmap_inc_factor ** (level - 1)),
                    out_channels=int(initial_fmaps * fmap_inc_factor**level),
                    kernel_sizes=kernel_sizes[level],
                    downsample_factor=downsample_factors[level],
                    activation=activation,
                    batch_norm=batch_norm,
                    group_norm=group_norm,
                    padding=padding,
                    padding_mode=padding_mode,
                    dropout=dropout,
                )
                for level in range(self.levels)
            ]
        )

        self.dims = len(kernel_sizes[0][0])


        def init_kaiming(m):
            if self.activation == nn.ReLU:
                nonlinearity = "relu"
            elif self.activation == nn.LeakyReLU:
                nonlinearity = "leaky_relu"
            else:
                raise ValueError(
                    f"Kaiming init not applicable for activation {self.activation}."
                )
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
                nn.init.zeros_(m.bias)

        if activation in (nn.ReLU, nn.LeakyReLU):
            self.apply(init_kaiming)
            logger.debug("Initialize conv weights with Kaiming init.")

    def forward(self, x):
        if x.ndim != self.dims + 2:
            raise ValueError(
                f"{x.ndim}D input given, {self.dims + 2}D input expected.")
        x = self.input_block(x)
        out = []
        for b in self.blocks:
            x = b(x)
            out += [x]
        return out


class ResNetBackbone(Resnet):
    """Backwards compatible.

    Both ``downsample_factors`` and ``kernel_sizes`` are expected to be a
    single-level tuples. 
    """

    def __init__(self, *args, **kwargs):
        kwargs["downsample_factors"] = tuple(
            (d, d) for d in kwargs["downsample_factors"])
        kwargs["kernel_sizes"] = (tuple(
            (k, k) for k in kwargs["kernel_sizes"]),)
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    import sys
    import torch
    t = torch.randn(2, 1, 256, 256)
    model = ResNetBackbone(in_channels=1, initial_fmaps=16, downsample_factors=[2 for _ in range(2)], kernel_sizes=[3 for _ in range(2)])
    print(f"Resolution levels: {model.levels}")
    print(f"Number of ResNet parameters: {sum(p.numel() for p in model.parameters())}")
    pred = model(t)
    for i, p in enumerate(pred):
        print(f"Level {i}: {p.shape}") 
    sys.exit(0)
