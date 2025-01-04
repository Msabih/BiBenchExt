
import sys
import argparse
import os
import os.path as osp
import numpy as np
import mmcv
import torch

from mmcv.runner import (
    load_checkpoint,
)
sys.path.append(r"./")

from bibench.models import build_architecture


def parse_args():
    parser = argparse.ArgumentParser(description='bibench test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--work-dir', help='the dir to save evaluation results')
    parser.add_argument('checkpoint', help='checkpoint file')

    
    args = parser.parse_args()
   
    return args


def main():
    sys.argv = [
    "python",  
    "test_models/dfsmn/bnn/dfsmn_bnn2.py",  # First argument (model script)
    "test_models/dfsmn/bnn/checkpoint.pth",  # Second argument (checkpoint path)
    "--work-dir", "results/result_no_compression"  # Argument for work directory
        ]

    args = parse_args()
 
    cfg = mmcv.Config.fromfile(args.config)

    # build the model and load checkpoint
    model = build_architecture(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model=model.eval()

    # # Define dummy input matching the model's input shape
    dummy_input = torch.randn(1, 1 ,32, 32)

    # Export to ONNX
    torch.onnx.export(
        model,              # The TorchScript model
        dummy_input,                 # Example input tensor
        "model.onnx",                # Output ONNX file
        export_params=True,          # Store trained weights inside the ONNX model
        opset_version=9,            # ONNX opset version (choose based on compatibility)
        input_names=["input"],       # Input layer name
        output_names=["output"],     # Output layer name
    )
 
    work_dir = osp.abspath(args.work_dir)
    onnx_to_tf_command = f'onnx2tf -i model.onnx -o {work_dir}'
    print( onnx_to_tf_command)

# Execute the command
    os.system(onnx_to_tf_command)


  

    

if __name__ == '__main__':
    main()

