_base_ = '../ablations/CBBSNet-MIT-B0.py'

model = dict(decode_head=dict(loss_decode=dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)))