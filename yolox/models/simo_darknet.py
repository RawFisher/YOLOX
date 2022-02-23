#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from torch import nn

from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck
from .network_blocks import get_activation


class SIMOCSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        # self.stem = nn.Conv2d(3, base_channels, 6, 2, 2)
        self.stem = SDCFocus(3, base_channels, ksize=6, stride=2, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        # self.dark5 = nn.Sequential(
        #     Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
        #     SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
        #     CSPLayer(
        #         base_channels * 16,
        #         base_channels * 16,
        #         n=base_depth,
        #         shortcut=False,
        #         depthwise=depthwise,
        #         act=act,
        #     ),
        # )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        # x = self.dark5(x)
        # outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class SDCFocus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.focus_conv = nn.Conv2d(in_channels, out_channels, ksize, stride, 2)
        self.focus_bn = nn.BatchNorm2d(out_channels)
        self.focus_act = get_activation(act, inplace=True)
        # self.conv = BaseConv(in_channels * 4, out_channels, 3, 1, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        x = self.focus_act(self.focus_bn(self.focus_conv(x)))
        return x
