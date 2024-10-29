_base_ = [
    '../_base_/models/vgg11.py', '../_base_/datasets/svhn.py',
    '../_base_/schedules/adam_1e-4_cosinelr_2e.py','../_base_/default_runtime.py'
]
model = dict(arch=dict(
    backbone=dict( depth=11, num_classes=10,
                  conv_cfg=dict(type='CBNNConv'),
                  norm_cfg =dict(type='BN', requires_grad=True),
                act_cfg=dict(type='Hardtanh', inplace=True),
                first_act_cfg=dict(type='Hardtanh', inplace=True)
                ))
                ,init_cfg=dict(type='Pretrained', checkpoint='data/pretrained/vgg11_svhn/bnn.pth'))