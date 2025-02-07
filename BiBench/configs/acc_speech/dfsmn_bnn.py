_base_ = [
    '../_base_/models/dfsmn.py', '../_base_/datasets/speech_commands - Copy.py',
    '../_base_/schedules/adam_1e-3_cosinelr_200e.py', '../_base_/default_runtime.py'
]

model = dict(
    arch=dict(
        backbone=dict(
            #first_conv_cfg=dict(type='BNNConv2d'),
            conv2d_cfg=dict(type='BNNConv2d'),
            conv1d_cfg=dict(type='BNNConv1d'),
            act_cfg=dict(type='PReLU')
        ),
        init_cfg=dict(type='Pretrained', checkpoint='data/Pretrained/Google speech commands/epoch_200_fp.pth')
    )
)
