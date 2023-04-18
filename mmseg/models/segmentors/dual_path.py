# Copyright (c) CVLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
import time



@SEGMENTORS.register_module()
class DualPath(BaseSegmentor):
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
        super(DualPath, self).__init__(init_cfg)
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

        assert self.with_decode_head

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

    def _init_cls_head(self, cls_head):
        self.cls_head = builder.build_head(cls_head)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img_batch, img_metas):
        x = self.extract_feat(img_batch)
        cls_out = self._cls_head_forward_test(x)
        img_label = torch.argmax(cls_out, dim=1)
        defect_index = torch.where(img_label > 0)[0]
        # print(defect_index)
        x = [a[defect_index] for a in x]
        seg_out = self._decode_head_forward_test(x, img_metas)
        seg_out = resize(
            input=seg_out,
            size=img_batch.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return defect_index, seg_out

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

    def _cls_head_forward_train(self, inputs, gt_category_label):
        losses = dict()
        loss_cls = self.cls_head.forward_train(inputs, gt_category_label)
        losses.update(add_prefix(loss_cls, 'cls'))
        return losses

    def _cls_head_forward_test(self, inputs):
        cls_logit = self.cls_head.forward_test(inputs)
        return cls_logit

    def forward_dummy(self, img):
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def get_category_label(self, gt_semantic_seg):
        gt_category_label = gt_semantic_seg.clone().detach()
        gt_category_label = gt_category_label.sum(dim=(2, 3))
        gt_category_label[gt_category_label > 0] = 1
        gt_category_label.squeeze_()
        return gt_category_label

    def forward_train(self, img, img_metas, gt_semantic_seg):
        gt_category_label = self.get_category_label(gt_semantic_seg)
        x = self.extract_feat(img)
        losses = dict()
        loss_cls = self._cls_head_forward_train(x, gt_category_label)
        losses.update(loss_cls)
        if gt_category_label.sum() > 0:
            x = [a[gt_category_label > 0] for a in x]# train seg-path with defective patches only.
            gt_semantic_seg = gt_semantic_seg[gt_category_label > 0]
            loss_decode = self._decode_head_forward_train(x, img_metas,
                                                          gt_semantic_seg)
            losses.update(loss_decode)
            if self.with_auxiliary_head:
                loss_aux = self._auxiliary_head_forward_train(
                    x, img_metas, gt_semantic_seg)
                losses.update(loss_aux)

        return losses

    def inference(self, img, img_meta, rescale):
        crop_list = []
        for i in range(8):
            for j in range(8):
                h1, w1 = 256 * i, 304 * j
                h2, w2 = h1 + 256, w1 + 304
                img_crop = img[:, :, h1:h2, w1:w2]
                crop_list.append(img_crop)
        img_batch = torch.cat(crop_list, dim=0)
        # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        defect_index, seg_logit = self.encode_decode(img_batch, img_meta)
        # print(prof.table())
        # prof.export_chrome_trace('./resnet_profile.json')
        if self.out_channels == 1:
            seg_out = F.sigmoid(seg_logit)
        else:
            seg_out = F.softmax(seg_logit, dim=1)

        flip = img_meta[0]['flip']

        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                seg_out = seg_out.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                seg_out = seg_out.flip(dims=(2, ))
        if self.out_channels == 1:
            seg_pred = (seg_out >
                        self.decode_head.threshold).to(seg_out).squeeze(1)
        else:
            seg_pred = seg_out.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_map = np.zeros((img.size(0), img.size(2), img.size(3)), dtype=np.uint8)
        for id, index in enumerate(defect_index):
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