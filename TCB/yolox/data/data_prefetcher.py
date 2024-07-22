#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.distributed as dist

from yolox.utils import synchronize

import random


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)      # MosaicDetection.__getitem__(idx)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.img_info, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.img_info = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            if(type(self.next_target)==type([1,2])):
                self.next_target = torch.cat(self.next_target,0)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input     # img: [1, 3, 800, 1440], e.g. [batchsize, channel, H, W]
        target = self.next_target   # annots: [1, 1000, 6], e,g, [batchsize, 1000, class_id + tlwh + track_id]
        img_info = self.img_info
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target,img_info

    def _input_cuda_for_image(self):
        if(type(self.next_input)==type([1,2])):
            self.next_input = torch.cat(self.next_input,0)
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


def random_resize(data_loader, exp, epoch, rank, is_distributed):
    tensor = torch.LongTensor(1).cuda()
    if is_distributed:
        synchronize()

    if rank == 0:
        if epoch > exp.max_epoch - 10:
            size = exp.input_size
        else:
            size = random.randint(*exp.random_size)
            size = int(32 * size)
        tensor.fill_(size)

    if is_distributed:
        synchronize()
        dist.broadcast(tensor, 0)

    input_size = data_loader.change_input_dim(multiple=tensor.item(), random_range=None)
    return input_size