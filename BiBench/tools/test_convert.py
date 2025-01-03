import sys
import argparse
import os
import os.path as osp
import numpy as np
import mmcv
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)
sys.path.append(r"./")
# sys.path.append(r"../")
# sys.path.append(r"D:\study\S3\Project thesis\Work\BiBenchExt\Libraries\BiNLP")

from bibench.apis import multi_gpu_test, single_gpu_test
from bibench.datasets import build_dataloader, build_dataset
from bibench.models import build_architecture


def parse_args():
    parser = argparse.ArgumentParser(description='bibench test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--work-dir', help='the dir to save evaluation results')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default='pa-mpjpe',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "pa-mpjpe" for H36M')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        default={},
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.val.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.val)
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=False)

    # build the model and load checkpoint
    model = build_architecture(cfg.model)

    
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model2=model

    # if not distributed:
    #     if args.device == 'cpu':
    #         model = model.cpu()
    #     else:
    #         model = MMDataParallel(model, device_ids=[0])
    #     outputs = single_gpu_test(model, data_loader)
    # else:
    #     model = MMDistributedDataParallel(
    #         model.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False)
    #     outputs = multi_gpu_test(model, data_loader, args.tmpdir,
    #                              args.gpu_collect)

    # rank, _ = get_dist_info()
    # eval_cfg = cfg.get('evaluation', args.eval_options)
    # eval_cfg.update(dict(metric=args.metrics))
    # if rank == 0:
    #     mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
    #     results = dataset.evaluate(outputs)
    #     for k, v in results.items():
    #         print(f'\n{k} : {v:.2f}')

    # if args.out and rank == 0:
    #     print(f'\nwriting results to {args.out}')
    #     mmcv.dump(results, args.out)

    model2.eval().to("cpu")
    dummy_input = torch.randn(1, 1 ,32, 32)

    # Export to ONNX
    torch.onnx.export(
        model2,              # The TorchScript model
        dummy_input,                 # Example input tensor
        "model.onnx",                # Output ONNX file
        export_params=True,          # Store trained weights inside the ONNX model
        opset_version=9,            # ONNX opset version (choose based on compatibility)
        input_names=["input"],       # Input layer name
        output_names=["output"],     # Output layer name
        #training=True,
        training=torch.onnx.TrainingMode.PRESERVE
        # do_constant_folding=False
    )
    import onnx
    model = onnx.load("model.onnx")

# # Freeze BN nodes (e.g., by adding unique names or attributes)
#     for node in model.graph.node:
#         if node.op_type == "Conv":
#            node.attribute.append(
#             onnx.helper.make_attribute("keep", True)
#         )
#     onnx.save(model, "model_locked_bn.onnx")
 
    work_dir = osp.abspath(args.work_dir)
    onnx_to_tf_command = f'onnx2tf -coto -nuo -osd -i model.onnx -o {work_dir}'
#     onnx_to_tf_command = f'onnx2tf -coto   -dgc -i model_locked_bn.onnx -o {work_dir}'
#     onnx_to_tf_command = f'onnx2tf -coto -i model_locked_bn.onnx -o {work_dir}'
#     onnx2tf -coto -i model_locked_bn.onnx -oh5

# # Execute the command
    os.system(onnx_to_tf_command)




if __name__ == '__main__':
    main()
