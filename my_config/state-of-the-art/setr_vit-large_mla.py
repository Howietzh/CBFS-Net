_base_ = [
    '../_base_/models/setr_mla.py',
    '../_base_/datasets/ccm.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    pretrained=None,
    backbone=dict(
        drop_rate=0),
    test_cfg=dict(mode='slide', crop_size=(256, 304), stride=(256, 304)))

optimizer = dict(
    lr=0.002,
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))
data = dict(samples_per_gpu=8, workers_per_gpu=8)
