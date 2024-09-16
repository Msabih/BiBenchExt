_base_ = [
    '../_base_/models/vgg11.py', '../_base_/datasets/svhn.py',
    '../_base_/schedules/sgd_1e-1_cosinelr_200e.py','../_base_/default_runtime.py'
]
model = dict(arch=dict(
    backbone=dict( depth=11, num_classes=10,
    norm_cfg =dict(type='BN', requires_grad=True))))