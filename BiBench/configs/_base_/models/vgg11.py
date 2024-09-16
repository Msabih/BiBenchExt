# model settings
arch = dict(
    repo='mmcls',
    type='ImageClassifier',
    backbone=dict(type='BiBench_VGG', depth=11, num_classes=200),
    #neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
model = dict(
    type='SimpleArchitecture',
    arch=arch
)
