import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from ..utils import UpConvBlock
from mmseg.ops import resize
from ..builder import NECKS
from mmseg.models.backbones.unet import BasicConvBlock

@NECKS.register_module()
class UNetNeck(BaseModule):
    def __init__(self,
                 in_channels=[64, 128, 256, 512, 1024],
                 num_convs=(2, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(UNetNeck, self).__init__(init_cfg)
        assert len(num_convs) == len(in_channels) - 1
        assert len(dilations) == len(num_convs)
        self.num_stages = len(num_convs)
        self.upconvs = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.in_channels = in_channels
        self.out_channels = in_channels[::-1]
        for i in range(self.num_stages):
            self.upconvs.append(
                ConvModule(
                    in_channels=self.out_channels[i],
                    out_channels=self.out_channels[i+1],
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                ))
            self.convs.append(
                BasicConvBlock(
                    in_channels=self.out_channels[i+1]*2,
                    out_channels=self.out_channels[i+1],
                    num_convs=num_convs[i],
                    stride=1,
                    dilation=dilations[i],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                ))
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = inputs[::-1]
        outs = [inputs[0]]
        for i in range(self.num_stages):
            upsample = resize(inputs[i], inputs[i+1].shape[2:], mode='bilinear')
            upsample = self.upconvs[i](upsample)
            cat_input = torch.cat([upsample, inputs[i+1]], dim=1)
            out = self.convs[i](cat_input)
            outs.append(out)
        return tuple(outs)

