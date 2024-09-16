_base_ = [
    '../_base_/models/mobilenet_v2_minist.py', '../_base_/datasets/minist.py',
    '../_base_/schedules/sgd_1e-1_cosinelr_200e.py', '../_base_/default_runtime.py'
]

model = dict(arch=dict(
    head=dict(type='LinearClsHead',
        num_classes=10,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),)))