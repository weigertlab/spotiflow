"""
Adapted from
https://github.com/bentaculum/backbones/blob/main/backbones/unet_2d.py
"""

import logging
import itertools
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """ConvBlock.

    Args:

        in_channels (``int``):

            The number of input channels for the first convolution in the
            block.

        out_channels (``int``):

            The number of output channels for all convolutions in the
            block.

        kernel_sizes (``tuple`` of ``tuple``):

            The number of tuples determines the number of
            convolutional layers in each ConvBlock. If not given, each
            ConvBlock will consist of two 3x3 convolutions.

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
        kernel_sizes,
        activation=nn.LeakyReLU,
        batch_norm=False,
        group_norm=False,
        padding=0,
        padding_mode='replicate'
    ):

        super().__init__()

        layers = []

        for k in kernel_sizes:
            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=k,
                padding=padding,
                padding_mode=padding_mode,
            ))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            if isinstance(group_norm, int) and group_norm > 0:
                layers.append(nn.GroupNorm(
                    num_groups=group_norm,
                    num_channels=out_channels,
                ))
            try:
                layers.append(activation(inplace=True))
            except TypeError:
                layers.append(activation())

            in_channels = out_channels

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):

        return self.conv_block(x)


class MaxPool2dDividing(nn.MaxPool2d):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        assert self.padding == 0
        assert self.kernel_size == self.stride
        assert self.dilation == 1

    def forward(self, input):
        if torch.any(torch.tensor(input.shape[-2:], device=input.device) %
                     torch.tensor(self.kernel_size, device=input.device) != 0):
            raise ValueError((
                f"Kernel size of {self} does not divide input of shape "
                f"{input.shape} "
            ))

        return super().forward(input)


class UNetBackbone(nn.Module):
    """Unet for square 2d inputs with isotropic operations.

    Input tensors are expected to be of shape ``(B, C, H, W)``.
    This model includes a 1x1Conv-head to return the desired
    number of out_channels.

    Args:

        in_channels:

            The number of input channels.

        inital_fmaps:

            The number of feature maps in the first layer. This is also the
            number of output feature maps. Stored in the ``channels``
            dimension.

        fmap_inc_factor:

            By how much to multiply the number of feature maps between
            layers. If layer 0 has ``k`` feature maps, layer ``l`` will
            have ``k*fmap_inc_factor**l``.

        downsample_factors:

            Tuple of tuples ``(x, y)`` to use to down- and up-sample the
            feature maps between layers.

        out_channels:

            The number of output_channels of the head.

        kernel_sizes (optional):

            Tuple of tuples. The number of tuples determines the number of
            convolutional layers in each ConvBlock. If not given, each
            ConvBlock will consist of two 3x3 convolutions.

        activation (``torch.nn.Module):

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

            `torch.nn.Conv2d` padding modes: `zeros`, `reflect`, `replicate` or
            `circular`.

        pad_input (``bool``):

            If set to ``True``, input tensors are padded to the smallest
            feasible input size using `padding_mode`. Only implemented for
            same padding, where it's ensured that input size equals output size.

        crop_input (``bool``):

            If set to ``True``, input tensors are cropped to the next
            feasible input size.
    """

    def __init__(
        self,
        in_channels,
        initial_fmaps,
        fmap_inc_factor=2,
        downsample_factors=((2, 2), (2, 2), (2, 2)),
        kernel_sizes=((3, 3), (3, 3), (3, 3)),
        activation=nn.LeakyReLU,
        constant_upsample=True,
        batch_norm=False,
        group_norm=False,
        padding=0,
        padding_mode='replicate',
        pad_input=False,
        crop_input=False,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.initial_fmaps = initial_fmaps

        if type(downsample_factors[0]) is int:
            downsample_factors = [(f, f) for f in downsample_factors]
 
        if type(kernel_sizes[0]) is int:
            kernel_sizes = [(k, k) for k in kernel_sizes]

        if not isinstance(fmap_inc_factor, int):
            raise ValueError(
                "Feature map increase factor has to be integer.")
        self.fmap_inc_factor = fmap_inc_factor

        for d in downsample_factors:
            if not isinstance(d, tuple):
                raise ValueError(
                    "Downsample factors have to be a list of tuples.")
           
            if not np.all(np.array(d) == d[0]):
                raise NotImplementedError(
                    "Anisotropic downsampling not implemented.")
        self.downsample_factors = downsample_factors

        for k in kernel_sizes:
            if not np.all(np.array(k) == k[0]):
                raise ValueError((
                    f"Anisotropic convolutional kernel {k} not supported."
                ))

        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.constant_upsample = constant_upsample

        if group_norm and batch_norm:
            raise ValueError("Do not apply multiple normalization approaches.")
        self.batch_norm = batch_norm
        if group_norm > initial_fmaps:
            raise ValueError(f"{group_norm=} bigger {initial_fmaps=}.")
        self.group_norm = group_norm

        self.padding = padding
        self.padding_mode = padding_mode
        self.pad_input = pad_input
        self.crop_input = crop_input

        if pad_input and \
                not all([all([(d - 1) / 2 == self.padding for d in k])
                         for k in kernel_sizes]):
            # pad_input only implemented for `same` padding
            raise NotImplementedError(
                "`pad_input` only implemented for `same` padding.")
        if pad_input and crop_input:
            raise ValueError(
                "`pad_input` and `crop_input` cannot both be `True`.")

        self.levels = len(downsample_factors) + 1

        self.out_channels_list = [initial_fmaps * fmap_inc_factor**level for level in range(self.levels)]

        # left convolutional passes
        self.l_conv = nn.ModuleList([
            ConvBlock(
                in_channels=in_channels if level == 0 else initial_fmaps *
                fmap_inc_factor**(level - 1),
                out_channels=initial_fmaps * fmap_inc_factor**level,
                kernel_sizes=kernel_sizes,
                activation=activation,
                batch_norm=batch_norm,
                group_norm=group_norm,
                padding=padding,
                padding_mode=padding_mode,
            )
            for level in range(self.levels)
        ])

        # left downsample layers
        self.l_down = nn.ModuleList([
            MaxPool2dDividing(
                kernel_size=downsample_factors[level],
                stride=downsample_factors[level]
            ) for level in range(self.levels - 1)
        ])

        # right upsample layers
        if constant_upsample:
            self.r_up = nn.ModuleList([
                nn.Upsample(
                    scale_factor=downsample_factors[level],
                    mode='nearest'
                )
                for level in range(self.levels - 1)
            ])
        else:
            self.r_up = nn.ModuleList([
                nn.ConvTranspose2d(
                    in_channels=initial_fmaps * fmap_inc_factor**(level + 1),
                    out_channels=initial_fmaps * fmap_inc_factor**(level + 1),
                    kernel_size=downsample_factors[level],
                    stride=downsample_factors[level],
                )
                for level in range(self.levels - 1)
            ])

        # right convolutional passes
        self.r_conv = nn.ModuleList([
            ConvBlock(
                in_channels=initial_fmaps * fmap_inc_factor**level +
                initial_fmaps * fmap_inc_factor**(level + 1),
                out_channels=initial_fmaps * fmap_inc_factor**level if level != 0 else initial_fmaps,
                kernel_sizes=kernel_sizes,
                activation=activation,
                batch_norm=batch_norm,
                group_norm=group_norm,
                padding=padding,
                padding_mode=padding_mode,
            )
            for level in range(self.levels - 1)
        ])


        # Initialize all Conv2d with Kaiming init
        def init_kaiming(m):
            if self.activation == nn.ReLU:
                nonlinearity = 'relu'
            elif self.activation == nn.LeakyReLU:
                nonlinearity = 'leaky_relu'
            else:
                raise ValueError(
                    f"Kaiming init not applicable for activation {self.activation}.")
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    nonlinearity=nonlinearity)
                nn.init.zeros_(m.bias)

        if activation in (nn.ReLU, nn.LeakyReLU):
            self.apply(init_kaiming)
            logger.debug("Initialize conv weights with Kaiming init.")

    def forward(self, x):

        x_spatial_shape = x.shape[2:]

        if self.pad_input:
            x = self._pad_input(x)
        if self.crop_input:
            x = self._crop_input(x)
        if self.padding == 0:
            if not self.is_valid_input_size(x.shape[2:]):
                raise ValueError((
                    f"Input size {x.shape[2:]} is not valid for"
                    " this Unet instance."
                ))

        if self.batch_norm and x.shape[0] == 1 and self.training:
            raise ValueError((
                "This Unet performs batch normalization, "
                "therefore inputs with batch size 1 are not allowed."
            ))
        
        out = self.iter_forward(x)
        # out,  = self.rec_forward(self.levels - 1, x)

        # if self.pad_input:
          #   # pad_input only implemented for same padding
          #   # assert in __init__
        #     out = self.crop(out, x_spatial_shape)

        return out
    
    def iter_forward(self, f_in):
        encoder_fmaps = []
        for lv in range(self.levels-1):
            f_in = self.l_conv[lv](f_in)
            f_in = self.l_down[lv](f_in)

            encoder_fmaps += [f_in]
        

        for lv in range(self.levels-1):
            f_in = self.r_up[self.levels - lv - 2](f_in)
            print(f_in.shape)
            cropped_enc_fmap = self.crop(encoder_fmaps[self.levels - lv - 2], f_in.shape[-2:])
            print(cropped_enc_fmap.shape)
            f_in = self.r_conv[lv](torch.cat([cropped_enc_fmap, f_in], dim=1))
            decoder_fmaps += [f_in]
        
        return decoder_fmaps[::-1]



    def rec_forward(self, level, f_in):

        # index of level in layer arrays
        i = self.levels - level - 1

        # convolve
        l_conv = self.l_conv[i](f_in)

        # end of recursion
        if level == 0:

            out = l_conv

        else:

            # down
            l_down = self.l_down[i](l_conv)

            # nested levels
            r_in = self.rec_forward(level - 1, l_down)

            # up
            r_up = self.r_up[i](r_in)

            # center crop l_conv
            l_conv_cropped = self.crop(l_conv, r_up.shape[-2:])

            # concat
            r_concat = torch.cat([l_conv_cropped, r_up], dim=1)

            # convolve
            out = self.r_conv[i](r_concat)

        return out

    def crop(self, x, shape):
        """Center-crop x to match spatial dimensions given by shape."""

        x_target_size = x.size()[:-2] + shape

        offset = tuple(
            (a - b) // 2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]

    def pad(self, x, shape):
        """Center-pad x to match spatial dimensions given by shape."""

        pad_total = tuple(
            a - b for a, b in zip(shape, x.size()[-2:])
        )

        pad_begin = tuple(
            t // 2 for t in pad_total
        )

        pad_end = tuple(
            (t + 1) // 2 for t in pad_total
        )
        # reverse axis order and interleave padding tuples
        # for torch.nn.functional.pad
        pad = tuple(itertools.chain(
            *zip(reversed(pad_begin), reversed(pad_end))))

        return torch.nn.functional.pad(
            input=x, pad=pad, mode=self.padding_mode)

    def _pad_input(self, x):
        target_shape = []
        for s in x.shape[-2:]:
            _s = s
            while not self.is_valid_input_size(_s):
                _s += 1
            target_shape.append(_s)

        return self.pad(x, tuple(target_shape))

    def _crop_input(self, x):
        target_shape = []
        for s in x.shape[-2:]:
            target_shape.append(self.valid_input_sizes_seq(s)[-1])

        return self.crop(x, tuple(target_shape))

    def valid_input_sizes_seq(self, n):

        sizes = []
        for i in range(n + 1):
            if self.is_valid_input_size(i):
                sizes.append(i)

        return sizes

    def is_valid_input_size(self, size):

        assert np.all(np.array(self.kernel_sizes) % 2 == 1)

        size = np.array(size, dtype=np.int_)
        ds_factors = [np.array(x, dtype=np.int_)
                      for x in self.downsample_factors]
        kernel_sizes = np.array(self.kernel_sizes, dtype=np.int_)
        p = self.padding

        def rec(level, s):
            # index of level in layer arrays
            i = self.levels - level - 1
            for k in kernel_sizes:
                s = s - (k - 1) + 2 * p
                if np.any(s < 1):
                    return False
            if level == 0:
                return s
            else:
                # down
                if np.any(s % ds_factors[i] != 0):
                    return False
                s = s // ds_factors[i]
                s = rec(level - 1, s)
                # up
                s = s * ds_factors[i]
                for k in kernel_sizes:
                    s = s - (k - 1) + 2 * p
                    if np.any(s < 1):
                        return False

            return s

        out = rec(self.levels - 1, size)
        if out is not False:
            out = True

        return out