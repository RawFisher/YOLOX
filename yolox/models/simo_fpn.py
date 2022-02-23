#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .sdc_darknet import SDCCSPDarknet
# from .simo_darknet import SIMOCSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv
from .network_blocks import get_activation


class SIMOFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark5",),
        in_channels=[1024,],
        encode_channels=[256, 256, 256],
        out_channels=[256, 256, 256],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = SDCCSPDarknet(depth, width, depthwise=depthwise, act=act)
        # self.backbone = SIMOCSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encode_channels = encode_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.align_layers = nn.ModuleList()
        for idx in range(len(self.in_channels)):
            self.align_layers.append(
                Conv(int(self.in_channels[idx] * width), int(self.encode_channels[idx] * width), 1, 1, act=act)
            )

        # bottom-up conv
        self.level_conv2_layers = nn.ModuleList()
        for idx in range(len(self.out_channels)):
            self.level_conv2_layers.append(
                Conv(int(self.encode_channels[idx] * width), int(self.encode_channels[idx] * width), 3, 1, act=act)
            )

        # extra layers
        self.extra_lvl_in_conv = ExtraConv(
            int(self.encode_channels[0] * width), int(self.encode_channels[0] * width), 3, 2, act=act
        )
        self.top_down_blocks = ExtraConv(
            int(self.encode_channels[0] * width), int(self.encode_channels[0] * width), 3, 2, act=act
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [align(out_features[f]) for f, align in zip(self.in_features, self.align_layers)]
        [C5] = features
        P5 = C5
        P4 = self.upsample(P5)
        P3 = self.upsample(P4)

        P5 = self.level_conv2_layers[0](P5)
        P4 = self.level_conv2_layers[1](P4)
        P3 = self.level_conv2_layers[2](P3)

        # extra layers
        P6 = self.extra_lvl_in_conv(C5) + self.top_down_blocks(P5)

        outputs = (P3, P4, P5, P6)
        return outputs


class ExtraConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        pad = ksize // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))