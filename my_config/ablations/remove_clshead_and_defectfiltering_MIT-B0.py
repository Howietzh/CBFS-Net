_base_ = 'CBBSNet-MIT-B0.py'

model = dict(
    cls_head=None,
    test_cfg=dict(mode='slide', crop_size=(256, 304), stride=(256, 304)),
    with_feature_filtering=False
)