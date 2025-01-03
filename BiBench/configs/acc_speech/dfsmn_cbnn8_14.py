_base_ = [
    '../_base_/models/dfsmn.py', '../_base_/datasets/speech_commands - Copy.py',
    '../_base_/schedules/adam_1e-4_cosinelr_50e.py', '../_base_/default_runtime.py'
]

model = dict(
    arch=dict(
        backbone=dict(
            #first_conv_cfg=dict(type='CBNNConv2d',k_bits=14),
            conv2d_cfg=dict(type='CBNNConv2d',k_bits=14),
            conv1d_cfg=dict(type='CBNNConv1d',k_bits=14),
            act_cfg=dict(type='Hardtanh')
        ),
        init_cfg=dict(type='Pretrained', checkpoint='data/Pretrained/Google speech commands/epoch_200.pth')
    )
)
