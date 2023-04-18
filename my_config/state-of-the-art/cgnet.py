_base_ = [
    '../_base_/models/cgnet.py', '../_base_/datasets/ccm.py',
    '../_base_/default_runtime.py']

# optimizer
optimizer = dict(type='Adam', lr=0.001, eps=1e-08, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
total_iters = 160000
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16000, metric=['mIoU', 'mFscore'], pre_eval=True)