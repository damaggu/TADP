# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from models.segmentation.deeplabv3plus_net.backbone import build_backbone
from models.segmentation.deeplabv3plus_net.ASPP import ASPP


class deeplabv3plus(nn.Module):

    def __init__(self, cfg):
        super(deeplabv3plus, self).__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = 2048
        self.aspp = ASPP(dim_in=input_channel,
                         dim_out=cfg.MODEL_ASPP_OUTDIM,
                         rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                         bn_mom=cfg.BN_MOM)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE // 4)

        indim = 256
        self.shortcut_conv = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, 1, 1, bias=True)),
                ('bn1', torch.nn.BatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.BN_MOM)),
                ('relu1', nn.ReLU(inplace=True)),
            ])
        )
        self.cat_conv = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(cfg.MODEL_ASPP_OUTDIM + cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1,
                                     padding=1, bias=True)),
                ('bn1', torch.nn.BatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.BN_MOM)),
                ('relu1', nn.ReLU(inplace=True)),
                ('dropout1', nn.Dropout(0.5)),
                ('conv2', nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1, bias=True)),
                ('bn2', torch.nn.BatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.BN_MOM)),
                ('relu2', nn.ReLU(inplace=True)),
                ('dropout2', nn.Dropout(0.1)),
            ])
        )
        self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()

    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = self.upsample4(result)
        return result
