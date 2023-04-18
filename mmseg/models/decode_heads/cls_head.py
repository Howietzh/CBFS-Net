# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from abc import ABCMeta, abstractmethod
from ..builder import build_loss
from mmcls.models.losses import Accuracy
import warnings
import torch.nn.functional as F

@HEADS.register_module()
class ClsHead(BaseModule, metaclass=ABCMeta):
    """classification head.

    Args:
        loss (dict): Config of classification loss.
        topk (int | tuple): Top-k accuracy.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use Mixup/CutMix or something like that during training,
            it is not reasonable to calculate accuracy. Defaults to False.
    """

    def __init__(self,
                 in_channels,
                 num_classes=2,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, ),
                 cal_acc=False,
                 init_cfg=None):
        super(ClsHead, self).__init__(init_cfg=init_cfg)

        assert isinstance(loss, dict)
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(self.in_channels, self.in_channels // 2)
        # self.fc2 = nn.Linear(self.in_channels // 2, self.num_classes)
        self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.compute_loss = build_loss(loss)
        self.compute_accuracy = Accuracy(topk=self.topk)
        self.cal_acc = cal_acc

    def loss(self, cls_score, gt_label, **kwargs):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(
            cls_score, gt_label, avg_factor=num_samples, **kwargs)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses['cls_accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }
        losses['cls_loss'] = loss
        return losses

    def forward_train(self, inputs, gt_category_label, **kwargs):
        cls_score = self(inputs)

        losses = self.loss(cls_score, gt_category_label, **kwargs)
        return losses

    def forward_test(self, inputs):

        return self.forward(inputs)

    def pre_logits(self, x):
        if isinstance(x, list):
            x = x[-1]
        return x

    def simple_test(self, cls_score, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            cls_score (tuple[Tensor]): The input classification score logits.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def post_process(self, pred):
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            outs = [self.gap(x) for x in inputs]
            outs = [out.view(x.size(0), -1) for out, x in zip(outs, inputs)]
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be list or torch.tensor')
        out = self.pre_logits(outs)
        # out = self.fc1(out)
        cls_score = self.fc(out)
        return cls_score


