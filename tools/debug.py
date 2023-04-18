
import torch
from mmseg.models.backbones import MobileNetV3, UNet, ResNetV1c, UNetEncoder, ResNeSt, ResNeXt, MixVisionTransformer
from mmseg.models.necks import UNetNeck, UNetDecoder
import cv2
import os

if __name__ == '__main__':

    input = torch.randn((64, 3, 256, 304))
    backbone = UNet(
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False)
    outs = backbone(input)
    for out in outs:
        print(out.shape)
    # outs = neck(outs)
    # for out in outs:
    #     print(out.shape)
