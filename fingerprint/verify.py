#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import math
import os
import os.path as osp

import torch
import torch.nn.functional as F
from torch.quantization import quantize_dynamic, prepare, convert

from fingerprint.victim.blackbox import Blackbox

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))



def quantize_model(model, quant_type='int8', backend='fbgemm', calib_data=None):
    """
    通用模型量化方法
    Args:
        model: 待量化的CNN模型
        quant_type: 'int8'或'float16'
        backend: 'fbgemm'(CPU)或'qnnpack'(ARM)
        calib_data: 静态量化校准数据(形状需匹配模型输入)
    Returns:
        量化后的模型
    """
    assert quant_type in ['int8', 'float16'], "仅支持int8/float16"
    model.eval()

    if quant_type == 'float16':
        return model.half()  # FP16直接转换

    # INT8量化逻辑
    if calib_data is None:
        # 动态量化(仅量化Linear/Conv层)
        return quantize_dynamic(
            model,
            {torch.nn.Conv2d, torch.nn.Linear},
            dtype=torch.qint8
        )
    else:
        # 静态量化流程
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        prepared = prepare(model)
        prepared(calib_data)  # 校准
        return convert(prepared)




def batch_forward(model, x, batch_size=200, argmax=False):
    """
    split x into batch_size, torch.cat result to return
    :param model:
    :param x:
    :param batch_size:
    :param argmax:
    :return:
    """
    device = next(model.parameters()).device
    steps = math.ceil(len(x) / batch_size)
    pred = []
    with torch.no_grad():
        for step in range(steps):
            off = int(step * batch_size)
            batch_x = x[off: off + batch_size].to(device)
            batch_y = model(batch_x).cpu().detach()
            print("batch_y:", batch_y)
            max_values, max_indices = torch.max(batch_y, dim=1)
            min_values, max_indices = torch.min(batch_y, dim=1)
            print("max(y):", max_values)
            print("min(y):", min_values)
            softmax_outputs = F.softmax(batch_y.float(), dim=1)
            print("softmax_outputs:", softmax_outputs)
            max_values, max_indices = torch.max(softmax_outputs, dim=1)
            print("max(softmax):", max_values)
            pred.append(batch_y)
    pred = torch.cat(pred)
    return pred.argmax(dim=1) if argmax else pred


def main():
    parser = argparse.ArgumentParser(description="Build micro benchmark.")
    parser.add_argument('victim_model_dir', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('fingerprint_dir', type=str, help='Destination directory to store fingerprint')
    parser.add_argument('--device_id', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('--transform_type', type=str, help='imagenet  cifar  tinyimagenet')
    parser.add_argument('--fingerprint_count', type=int, default=100, help='images loaded count')
    parser.add_argument("--quantization", default="no", type=str, help="int8 half no", choices=["int8","half","no"])

    args, unknown = parser.parse_known_args()
    args.ROOT = ROOT
    args = parser.parse_args()
    params = vars(args)

    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    blackbox_dir = params['victim_model_dir']
    model = Blackbox.from_modeldir(blackbox_dir, device).get_model()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    out_path = params['fingerprint_dir']
    quantization = params['quantization']
    dest_image_path = osp.join(out_path, 'data.pt')
    fingerprints_data = torch.load(dest_image_path)
    print("-> verify ownership...")
    test_x, test_y = fingerprints_data["test_x"], fingerprints_data["test_y"]
    y1, y2 = test_y.cpu(), []
    with torch.no_grad():
        if quantization == 'no':
            y2.append(batch_forward(model=model, x=test_x, argmax=True))
        elif quantization == 'int8':
            device = torch.device('cpu')
            model.to(device)
            quantized_model = quantize_model(model, quant_type='int8')
            y2.append(batch_forward(model=quantized_model, x=test_x, argmax=True))
        elif quantization == 'half':
            quantized_model = quantize_model(model, quant_type='float16')
            y2.append(batch_forward(model=quantized_model, x=test_x.half().to(device), argmax=True))
    y2 = torch.cat(y2)
    matching_rate = round(float(y1.eq(y2.view_as(y1)).sum()) / len(y1), 5)
    print(f"matching_rate:{matching_rate}")


if __name__ == "__main__":
    main()























