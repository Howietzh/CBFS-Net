# dataset settings
dataset_type = 'CCMDataset'
train_data_root = 'data/CCM_windowed/'
val_data_root = 'data/CCM_original/'
img_norm_cfg = dict(
    mean=[73.978386, 72.509575, 71.09528], std=[33.00916, 28.699066, 26.370403], to_rgb=True)
crop_size = (256, 304)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(304, 256), ratio_range=(0.75, 1.25), keep_ratio=False),
    dict(type='RandomRotate', prob=0.5, degree=45.0, auto_bound=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2432, 2048),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=16, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=train_data_root,
        split='train.txt',
        img_dir='images/train',
        ann_dir='annotations/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=val_data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=val_data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        pipeline=test_pipeline))
