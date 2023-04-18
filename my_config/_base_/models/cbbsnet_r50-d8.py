norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DPBSN',
    pretrained=None,
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    cls_head=dict(
        type='ClsHead',
        in_channels=2048,
        num_classes=2,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        topk=(1, ),
        cal_acc=False,
        init_cfg=None),
    train_cfg=dict())