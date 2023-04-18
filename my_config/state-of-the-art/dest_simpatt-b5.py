_base_ = [
    '../_base_/models/dest_simpatt-b0.py',
    '../_base_/datasets/ccm.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
embed_dims = [64, 128, 250, 320]
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(embed_dims=embed_dims, num_layers=[3, 10, 16, 5]),
    decode_head=dict(in_channels=embed_dims, channels=64),
    test_cfg=dict(mode='slide', crop_size=(256, 304), stride=(256, 304)))
custom_imports = dict(imports=['projects.dest.models'])

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=1.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=8, workers_per_gpu=8)
