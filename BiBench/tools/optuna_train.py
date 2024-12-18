import sys
import json
sys.path.append(r"./")
import argparse
import copy
import os
import io
import os.path as osp
import time
import optuna
from optuna.trial import TrialState
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from bibench import __version__
from bibench.apis import set_random_seed, train_model_op
from bibench.datasets import build_dataset
from bibench.models import build_architecture
from bibench.utils import collect_env, get_root_logger

def get_targets(model):
    targets=[]
    for k in model.state_dict().keys():
        if 'conv' in k or 'fc_trans.0' in k or 'fc_trans.4' in k or 'memory.0' in k or 'front_end.3' in k:
            name=k.split(".")
            del name[-1]
            name=".".join(name)
            if check_class(model, name):
                targets.append(name)
    targets=list(set(targets))
    return(targets)

def check_class(obj, attr_string):
    attrs = attr_string.split('.')
    for attr in attrs:
        # if attr.isdigit():
        #     obj=obj[int(attr)]
        # else:    
            obj = getattr(obj, attr)
    return str(type(obj))== "<class 'bibench.models.layers.custom_bnn.CBNNConv2d'>"  or  str(type(obj))== "<class 'bibench.models.layers.custom_bnn.CBNNConv1d'>"   


def set_nested_attr(obj, attr_string, value):
    attrs = attr_string.split('.')
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)

def size(state_dict):
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    size_in_MB = buffer.tell() / (1024 * 1024)  
    buffer.close()  
    print('model size: {:.3f}MB'.format(size_in_MB),)
    return size_in_MB

def define_model(trial,model,targets):
        model_copy = copy.deepcopy(model)
        for i,attr in enumerate(targets):
            k_bits = trial.suggest_int(f"k_bits{i+1}", 10, 16,step=2)
            set_nested_attr(model_copy, attr+".k_bits", k_bits)
            set_nested_attr(model_copy, attr+".first_iter",True)   
        return model_copy
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--device', help='device used for training')
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    model = build_architecture(cfg.model)
    
    model.init_weights()
    targets=get_targets(model)
  

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save bibench version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            bibench_version=__version__, config=cfg.pretty_text)
    # # add an attribute for visualization convenience
    def objective(trial):
        model2 = define_model(trial, model, targets)
        cfg_=copy.deepcopy(cfg)
        results=train_model_op(
            model2,
            datasets,
            cfg_,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            device='cpu' if args.device == 'cpu' else 'cuda',
            meta=meta)
        accuracy=results.last_val_accuracy["accuracy_top-1"]
        model_size=size(model2)
        return accuracy,model_size
    work_dir = osp.abspath(cfg.work_dir)
    save_path = osp.join(work_dir, 'targets.json')
    with open(save_path, 'w') as f:
        json.dump(targets, f)
    storage_path = f"sqlite:///{os.path.join(work_dir, 'db.sqlite3')}"
    study = optuna.create_study(directions=["maximize","minimize"],storage=storage_path,
    study_name="accuracy,size"
)
    study.optimize(objective, n_trials=50)

if __name__ == '__main__':
    main()
