_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/ccm.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
        num_classes=5),
    auxiliary_head=dict(
        in_channels=256,
        channels=64,
        num_classes=5),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(256, 304), stride=(256, 304)))

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
)