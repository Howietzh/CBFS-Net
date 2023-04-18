# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
import numpy as np
import mmcv
from mmseg.models.backbones.unet import BasicConvBlock


@SEGMENTORS.register_module()
class DPBSN(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 cls_head,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DPBSN, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_cls_head(cls_head)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # self.detail_branch = BasicConvBlock(
        #     in_channels=3,
        #     out_channels=16,
        #     num_convs=2,
        #     norm_cfg=dict(type='SyncBN', requires_grad=True)
        # )
        assert self.with_decode_head

    def _init_cls_head(self, cls_head):
        self.cls_head = builder.build_head(cls_head)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def batch_select(self, x, patch_label):
        """Select Defective Batches from [a list of] feature x"""
        self.defect_index = torch.where(patch_label > 0)[0]
        if isinstance(x, (list, tuple)):
            x = [a[self.defect_index] for a in x]
        else:
            x = x[self.defect_index]
        return x

    def extract_feat(self, imgs):
        x = self.backbone(imgs)
        # detail = self.detail_branch(imgs)
        # out = []
        # out.append(detail)
        # for item in x:
        #     out.append(item)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        cls_out = self._cls_head_forward_test(x)
        patch_pred = torch.argmax(cls_out, dim=1)
        patch_label = patch_pred > 0
        x = self.batch_select(x, patch_label)
        if self.with_neck:
            x = self.neck(x)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _cls_head_forward_train(self, x, gt_category_label):
        losses = dict()
        loss_cls = self.cls_head.forward_train(x, gt_category_label)
        losses.update(add_prefix(loss_cls, 'cls'))
        return losses

    def _cls_head_forward_test(self, x):
        cls_logit = self.cls_head.forward_test(x)
        return cls_logit

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def get_category_label(self, gt_semantic_seg):
        gt_category_label = gt_semantic_seg.clone().detach()
        gt_category_label = gt_category_label.sum(dim=(2, 3))
        gt_category_label[gt_category_label > 0] = 1
        gt_category_label.squeeze_()
        return gt_category_label

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)
        gt_category_label = self.get_category_label(gt_semantic_seg)

        losses = dict()
        loss_cls = self._cls_head_forward_train(x, gt_category_label)
        losses.update(loss_cls)

        x = self.batch_select(x, gt_category_label)
        if self.with_neck:
            x = self.neck(x)
        gt_semantic_seg = self.batch_select(gt_semantic_seg, gt_category_label)
        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    # TODO refactor

    def inference(self, img, img_meta, rescale):
        crop_list = []
        for i in range(8):
            for j in range(8):
                h1, w1 = 256 * i, 304 * j
                h2, w2 = h1 + 256, w1 + 304
                img_crop = img[:, :, h1:h2, w1:w2]
                crop_list.append(img_crop)
        img_batch = torch.cat(crop_list, dim=0)
        seg_logit = self.encode_decode(img_batch, img_meta)
        if self.out_channels == 1:
            seg_out = F.sigmoid(seg_logit)
        else:
            seg_out = F.softmax(seg_logit, dim=1)

        flip = img_meta[0]['flip']

        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                seg_out = seg_out.flip(dims=(3,))
            elif flip_direction == 'vertical':
                seg_out = seg_out.flip(dims=(2,))
        if self.out_channels == 1:
            seg_pred = (seg_out >
                        self.decode_head.threshold).to(seg_out).squeeze(1)
        else:
            seg_pred = seg_out.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_map = np.zeros((img.size(0), img.size(2), img.size(3)), dtype=np.uint8)
        for id, index in enumerate(self.defect_index):
            img_index = index // 64
            patch_index = index % 64
            h_index = patch_index // 8
            w_index = patch_index % 8
            h1, w1 = 256 * h_index, 304 * w_index
            h2, w2 = h1 + 256, w1 + 304
            seg_map[img_index, h1:h2, w1:w2] = seg_pred[id]
        return seg_map

    def simple_test(self, img, img_meta, rescale=True):
        seg_map = self.inference(img, img_meta, rescale)
        seg_map = list(seg_map)
        if rescale:
            seg_map = [mmcv.imresize(seg, img_meta[0]['ori_shape'][-2::-1], interpolation='nearest') for seg in seg_map]
        return seg_map

    def aug_test(self, img, img_meta, rescale=True):
        seg_map = self.inference(img, img_meta, rescale)
        seg_map = list(seg_map)
        if rescale:
            seg_map = [mmcv.imresize(seg, img_meta[0]['ori_shape'][-2::-1], interpolation='nearest') for seg in seg_map]
        return seg_map
