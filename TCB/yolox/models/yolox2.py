#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn
from .yolo_head2 import YOLOXHead2
from .yolo_pafpn import YOLOPAFPN

class YOLOX2(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()      # backbone, CSPNet with PANet
        if head is None:
            head = YOLOXHead2(1)        # head

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None, last_reid=None):
        # fpn output content features of [dark3, dark4, dark5]#x:[b,3,800,1440]
        if not self.training:# one element
            fpn_outs = self.backbone(x)#[1,320,100,180],[1,640,50,90],[1,1280,25,45]
            outputs = self.head(xin=fpn_outs,imgs=x,last_reid=last_reid)
            return outputs
        else:#train
            assert targets is not None
            batch_size = x.shape[0] // 2
            x1 = x[:batch_size,:]
            x2 = x[batch_size:,:]
            target1 = targets[:batch_size,:]
            target2 = targets[batch_size:,:]
            fpn_outs1 = self.backbone(x1)
            feature_reid,feature_id = self.head(xin=fpn_outs1,labels=target1,imgs=x1)
            fpn_outs2 = self.backbone(x2)
            total_loss,loss1,loss2= self.head(xin=fpn_outs2,labels=target2,imgs=x2,last_reid=feature_reid,feature_id=feature_id)
            outputs = {"total_loss": total_loss,"loss":loss1,"reid_loss":loss2}
            #print(1)
            return outputs
