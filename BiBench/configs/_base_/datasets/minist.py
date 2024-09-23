
# dataset settings
dataset_repo = 'mmcls'
dataset_type = 'MNIST'
img_norm_cfg = dict(mean=[33.46], std=[78.87], to_rgb=True)
train_pipeline = [
    dict(type='Resize', size=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='Resize', size=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=1,
    train=dict(
        repo=dataset_repo,type=dataset_type, data_prefix='data/datasets/MINST', pipeline=train_pipeline),
    val=dict(
        repo=dataset_repo,type=dataset_type, data_prefix='data/datasets/MINST', pipeline=test_pipeline,
        test_mode=True))
evaluation = dict(
    save_best='accuracy_top-1')

