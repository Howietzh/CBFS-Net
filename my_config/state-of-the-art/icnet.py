_base_ = [
    '../_base_/models/icnet_r50-d8.py', '../_base_/datasets/ccm.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (1024, 1024)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
)

# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# # optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=0.0005)
# lr_config = dict(policy='poly', power=2.0, min_lr=1e-5, by_epoch=False)
# checkpoint_config = dict(by_epoch=False, interval=16000)
# evaluation = dict(interval=16000, metric=['mIoU', 'mFscore'], pre_eval=True)