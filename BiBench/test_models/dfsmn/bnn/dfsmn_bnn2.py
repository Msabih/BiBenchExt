arch = dict(
    repo='bispeech',
    type='ImageClassifier',
    backbone=dict(
        type='Dfsmn',
        in_channels=1,
        n_mels=32,
        num_layer=8,
        frondend_channels=16,
        frondend_kernel_size=5,
        hidden_size=256,
        backbone_memory_size=128,
        left_kernel_size=2,
        right_kernel_size=2,
        dilation=1,
        drop_path_rate=0.0),
    head=dict(
        type='LinearClsHead',
        in_channels=1024,
        num_classes=12,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
model = dict(
    type='SimpleArchitecture',
    arch=dict(
        repo='bispeech',
        type='ImageClassifier',
        backbone=dict(
            type='Dfsmn',
            in_channels=1,
            n_mels=32,
            num_layer=8,
            frondend_channels=16,
            frondend_kernel_size=5,
            hidden_size=256,
            backbone_memory_size=128,
            left_kernel_size=2,
            right_kernel_size=2,
            dilation=1,
            drop_path_rate=0.0,
            conv2d_cfg=dict(type='BNNConv2d'),
            conv1d_cfg=dict(type='BNNConv1d'),
            act_cfg=dict(type='PReLU')),
        head=dict(
            type='LinearClsHead',
            in_channels=1024,
            num_classes=12,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0))))
dataset_repo = 'bispeech'
dataset_type = 'SpeechCommandDataset'
n_mels = 32
data_prefix = 'data/datasets'
num_classes = 12
version = 'speech_commands_v0.01'
train_pipeline = [
    dict(type='ChangeAmplitude'),
    dict(type='ChangeSpeedAndPitchAudio'),
    dict(type='TimeshiftAudio'),
    dict(type='FixAudioLength'),
    dict(
        type='MelSpectrogram',
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        n_mels=32,
        normalized=True),
    dict(type='AmplitudeToDB'),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='FixAudioLength'),
    dict(
        type='MelSpectrogram',
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        n_mels=32,
        normalized=True),
    dict(type='AmplitudeToDB'),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=96,
    workers_per_gpu=4,
    train=dict(
        repo='bispeech',
        type='SpeechCommandDataset',
        subset='training',
        data_prefix='data/datasets',
        noise_ratio=0.3,
        noise_max_scale=0.3,
        num_classes=12,
        version='speech_commands_v0.01',
        pipeline=[
            dict(type='ChangeAmplitude'),
            dict(type='ChangeSpeedAndPitchAudio'),
            dict(type='TimeshiftAudio'),
            dict(type='FixAudioLength'),
            dict(
                type='MelSpectrogram',
                sample_rate=16000,
                n_fft=2048,
                hop_length=512,
                n_mels=32,
                normalized=True),
            dict(type='AmplitudeToDB'),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        repo='bispeech',
        type='SpeechCommandDataset',
        subset='validation',
        data_prefix='data/datasets',
        num_classes=12,
        version='speech_commands_v0.01',
        pipeline=[
            dict(type='FixAudioLength'),
            dict(
                type='MelSpectrogram',
                sample_rate=16000,
                n_fft=2048,
                hop_length=512,
                n_mels=32,
                normalized=True),
            dict(type='AmplitudeToDB'),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        repo='bispeech',
        type='SpeechCommandDataset',
        subset='testing',
        data_prefix='data/datasets',
        num_classes=12,
        version='speech_commands_v0.01',
        pipeline=[
            dict(type='FixAudioLength'),
            dict(
                type='MelSpectrogram',
                sample_rate=16000,
                n_fft=2048,
                hop_length=512,
                n_mels=32,
                normalized=True),
            dict(type='AmplitudeToDB'),
            dict(type='Collect', keys=['img'])
        ]))
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0)
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(interval=1, save_last=True)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = 'D:\study\S3\Project thesis\Results\Speech detection\dfsmn_BNN\latest.pth'
workflow = [('train', 1)]
work_dir = 'D:\study\S3\Project thesis\Results\Speech detection\dfsmn_BNN'
gpu_ids = range(0, 1)
