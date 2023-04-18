_base_ = '../backbones/cbbsnet_mit-b0.py'

model = dict(
    decode_head=dict(
        loss_decode=dict(type='FocalLoss', gamma=2.0, alpha=0.5, use_sigmoid=True, loss_weight=1.0)
    )
)