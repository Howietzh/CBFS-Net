_base_ = [
    '../_base_/models/bisenetv2.py', '../_base_/datasets/ccm.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]


data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
)


checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16000, metric=['mIoU', 'mFscore'], pre_eval=True)