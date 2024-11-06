_base_ = [
    '../_base_/models/dfsmn.py', '../_base_/datasets/speech_commands - Copy.py',
    '../_base_/schedules/adam_1e-4_cosinelr_50e.py', '../_base_/default_runtime.py'
]

model = dict(
    arch=dict(
        backbone=dict(
            conv2d_cfg=dict(type='CBNNConv2d'),
            conv1d_cfg=dict(type='CBNNConv1d'),
            act_cfg=dict(type='PReLU')
        ),
        init_cfg=dict(type='Pretrained', checkpoint='data/Pretrained/Google speech commands/epoch_200.pth')
    )
)
