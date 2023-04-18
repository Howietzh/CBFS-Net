_base_ = 'CBBSNet-R-18-D8.py'

model = dict(
    test_cfg=dict(mode='slide', crop_size=(256, 304), stride=(256, 304))
)