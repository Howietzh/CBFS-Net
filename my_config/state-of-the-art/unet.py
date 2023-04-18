_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/ccm.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    decode_head=dict(num_classes=5),
    auxiliary_head=dict(num_classes=5),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(256, 304), stride=(256, 304)))

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
)