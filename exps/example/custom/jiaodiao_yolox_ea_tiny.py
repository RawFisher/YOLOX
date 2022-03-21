#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp.yolox_ea import YOLOXEAExp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.input_scale = (416, 416)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False

        # Define yourself dataset path
        self.data_dir = "datasets/jiaodiao"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        self.num_classes = 9

        self.max_epoch = 300
        self.data_num_workers = 8
        self.eval_interval = 1

        self.basic_lr_per_img = 0.02 / 64.0  # 0.01 / 64.0
        self.min_lr_ratio = 0.1  # 0.05

        self.print_interval = 100
