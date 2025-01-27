_base_ = [
    '../_base_/models/resnet34.py', '../_base_/datasets/cifar10.py',
    '../_base_/schedules/adam_1e-4_cosinelr_2e.py', '../_base_/default_runtime.py'
]

model = dict(
    arch=dict(
        backbone=dict(
            type='BiBench_ResNet_CIFAR',
            conv_cfg=dict(type='CBNNConv'),
            first_act_cfg=dict(type='Hardtanh', inplace=True),
            act_cfg=dict(type='Hardtanh', inplace=True)
        ),
        head=dict(num_classes=10)
      ),
    init_cfg=dict(type='Pretrained', checkpoint='data/pretrained/cifar10/bnn_epoch_185.pth')
)
