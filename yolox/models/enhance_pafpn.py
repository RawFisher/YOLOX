#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .sdc_darknet import SDCCSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv
from .network_blocks import get_activation


class EnhancePAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = SDCCSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        self.out_channels = self.in_channels[0]
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.align_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.align_layers.append(
                Conv(int(in_channels[idx] * width), int(self.out_channels * width), 1, 1, act=act)
            )
        self.C3_p4 = CSPLayer(
            int(2 * self.out_channels * width),
            int(self.out_channels * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.C3_p3 = CSPLayer(
            int(2 * self.out_channels * width),
            int(self.out_channels * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(self.out_channels * width), int(self.out_channels * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * self.out_channels * width),
            int(self.out_channels * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(self.out_channels * width), int(self.out_channels * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * self.out_channels * width),
            int(self.out_channels * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # extra layers
        self.extra_lvl_in_conv = ExtraConv(
            int(self.out_channels * width), int(self.out_channels * width), 3, 2, act=act
        )
        self.top_down_blocks = ExtraConv(
            int(self.out_channels * width), int(self.out_channels * width), 3, 2, act=act
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
        [C3, C4, C5] = features

        f_out0 = self.upsample(C5)  # 512/16
        f_out0 = torch.cat([f_out0, C4], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        f_out1 = self.upsample(f_out0)  # 256/8
        f_out1 = torch.cat([f_out1, C3], 1)  # 256->512/8
        P3 = self.C3_p3(f_out1)  # 512->256/8

        P4 = self.bu_conv2(P3)  # 256->256/16
        P4 = torch.cat([P4, f_out0], 1)  # 256->512/16
        P4 = self.C3_n3(P4)  # 512->512/16

        P5 = self.bu_conv1(P4)  # 512->512/32
        P5 = torch.cat([P5, C5], 1)  # 512->1024/32
        P5 = self.C3_n4(P5)  # 1024->1024/32

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