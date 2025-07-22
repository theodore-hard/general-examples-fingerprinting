#!/usr/bin/python
"""
compress model by prune quantization or distill
"""

import argparse
import json
import os
import os.path as osp
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic, prepare, convert
from torch.utils.data import DataLoader
import config as cfg
import utils.model as model_utils
import utils.utils as fingerprint_utils
import datasets
from victim.blackbox import Blackbox


def create_dir(dir_path):
    if not osp.exists(dir_path):
        print('Path {} does not exist. Creating it...'.format(dir_path))
        os.makedirs(dir_path)

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


def model_global_prune(detect_model, amount):
    a, b =0., 0.
    for p in detect_model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    sparsity = b / a
    print('%.3g global sparsity' % sparsity)

    parameters_to_prune = list()
    prune_nums = 0
    for i, modules in enumerate(detect_model.modules()):
        if isinstance(modules, nn.Conv2d):
            prune_nums += 1
            parameters_to_prune.append((modules, 'weight'))
        elif isinstance(modules, nn.Linear):
            prune_nums += 1
            parameters_to_prune.append((modules, 'weight'))
        elif isinstance(modules, nn.BatchNorm2d):
            prune_nums += 2
            parameters_to_prune.append((modules, 'weight'))
            parameters_to_prune.append((modules, 'bias'))
    print(parameters_to_prune)
    parameters_to_prune = tuple(parameters_to_prune)
    assert (prune_nums == len(parameters_to_prune))
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    remove_nums = 0
    for i, modules in enumerate(detect_model.modules()):
        if isinstance(modules, nn.Conv2d):
            remove_nums += 1
            prune.remove(modules, 'weight')
        elif isinstance(modules, nn.Linear):
            remove_nums += 1
            prune.remove(modules, 'weight')
        elif isinstance(modules, nn.BatchNorm2d):
            remove_nums += 2
            prune.remove(modules, 'weight')
            prune.remove(modules, 'bias')
    assert(prune_nums == remove_nums)

    a, b =0., 0.
    for p in detect_model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    sparsity = b / a
    print('%.3g global sparsity' % sparsity)

    return detect_model


def prune_layer(model, amount=0.2):
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)
            prune.remove(m, 'weight')
    a, b =0., 0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    sparsity = b / a
    print('%.3g global sparsity' % sparsity)


def main():
    parser = argparse.ArgumentParser(description='Extract model fingerprint')
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('out_dir', metavar='PATH', type=str,
                        help='Destination directory to store model')
    parser.add_argument('method', type=str, choices=('prune', 'quantization'))

    parser.add_argument('--test_dataset_name', type=str, help='test dataset name')
    parser.add_argument('--test_dataset_type', type=str, choices=('train', 'test', 'finetune'),
                        help='test dataset type')
    parser.add_argument('--amount', type=float, default='0.2')
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=2)
    args = parser.parse_args()
    params = vars(args)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    out_dir = params['out_dir']
    fingerprint_utils.create_dir(out_dir)
    blackbox_dir = params['victim_model_dir']
    blackbox = Blackbox.from_modeldir(blackbox_dir, device)
    victim_model = blackbox.get_model()
    victim_model.eval()

    valid_datasets = datasets.__dict__.keys()
    test_dataset_name = params['test_dataset_name']
    batch_size = params['batch_size']
    num_workers = params['num_workers']
    test_dataset = None
    if test_dataset_name is not None:
        if test_dataset_name not in valid_datasets:
            raise ValueError('Dataset {} not found. Valid arguments = {}'.format(test_dataset_name, valid_datasets))
        test_dataset_class = datasets.__dict__[test_dataset_name]
        test_dataset_modelfamily = datasets.dataset_to_modelfamily[test_dataset_name]
        test_transform = datasets.modelfamily_to_transforms[test_dataset_modelfamily]['test']
        test_dataset_type = params['test_dataset_type']
        test_dataset = test_dataset_class(test_dataset_type, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    compress_method = params['method']
    if compress_method == 'prune':
        #print('before prune, test accuracy:')
        #test_acc, test_topk_acc = model_utils.test_model(victim_model, test_loader, device=device)
        #torch.cuda.empty_cache()
        amount = params['amount']
        victim_model = model_global_prune(victim_model, amount)
        torch.cuda.empty_cache()
        print('after prune, test accuracy:')
        test_acc, test_topk_acc = model_utils.test_model(victim_model, test_loader, device=device)
        state = {
            'epoch': None,
            'arch': victim_model.__class__,
            'state_dict': victim_model.state_dict(),
            'best_acc': test_acc,
            'optimizer': None,
            'created_on': str(datetime.now()),
        }
        model_out_path = osp.join(out_dir, 'checkpoint.pth.tar')
        torch.save(state, model_out_path)

        params['model_arch'] = blackbox.model_arch
        params['num_classes'] = blackbox.num_classes
        params['created_on'] = str(datetime.now())
        params_out_path = osp.join(out_dir, 'params.json')
        with open(params_out_path, 'w') as jf:
            json.dump(params, jf, indent=True)
    elif compress_method == 'quantization':
        # for key in victim_model.parameters().keys():
        #     victim_model.parameters()[key] = victim_model.parameters()[key].half()
        print('start generate int8 quantized model')
        device = torch.device('cpu')
        victim_model.to(device)
        # sample= next(iter(test_loader))[0].to(device)
        # quantized_model = quantize_model(victim_model, quant_type='int8',  calib_data=sample)
        quantized_model = quantize_model(victim_model, quant_type='int8')
        print('test on test dataset:')
        test_acc, test_topk_acc = model_utils.test_model(quantized_model, test_loader, device=device)
        state = {
            'epoch': None,
            'arch': quantized_model.__class__,
            'state_dict': quantized_model.state_dict(),
            'best_acc': test_acc,
            'optimizer': None,
            'created_on': str(datetime.now()),
        }
        model_out_dir = osp.join(out_dir,'int8')
        create_dir(model_out_dir)
        model_out_path = osp.join(model_out_dir, 'checkpoint.pth.tar')
        torch.save(state, model_out_path)

        params['model_arch'] = blackbox.model_arch
        params['num_classes'] = blackbox.num_classes
        params['test_acc'] = test_acc
        params['created_on'] = str(datetime.now())
        params_out_path = osp.join(model_out_dir, 'params.json')
        with open(params_out_path, 'w') as jf:
            json.dump(params, jf, indent=True)

        print('start generate float16 quantized model')
        quantized_model = quantize_model(victim_model, quant_type='float16')
        print('test on test dataset:')
        device = torch.device('cuda')
        quantized_model.to(device)
        test_acc, test_topk_acc = model_utils.test_half_model(quantized_model, test_loader, device=device)
        state = {
            'epoch': None,
            'arch': quantized_model.__class__,
            'state_dict': quantized_model.state_dict(),
            'best_acc': test_acc,
            'optimizer': None,
            'created_on': str(datetime.now()),
        }
        model_out_dir = osp.join(out_dir,'float16')
        create_dir(model_out_dir)
        model_out_path = osp.join(model_out_dir, 'checkpoint.pth.tar')
        torch.save(state, model_out_path)

        params['model_arch'] = blackbox.model_arch
        params['num_classes'] = blackbox.num_classes
        params['created_on'] = str(datetime.now())
        params['test_acc'] = test_acc
        params_out_path = osp.join(model_out_dir, 'params.json')
        with open(params_out_path, 'w') as jf:
            json.dump(params, jf, indent=True)

    elif compress_method == 'distill':
        pass


if __name__ == '__main__':
    main()