#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import json
import os
import os.path as osp
from datetime import datetime

import torch
import torchvision

import fingerprint.utils.model as model_utils
from fingerprint import datasets
from fingerprint.victim.blackbox import Blackbox


def main():
    parser = argparse.ArgumentParser(description='finetune a model')
    # Required arguments
    parser.add_argument('train_dataset_name', type=str, help='train dataset name')
    parser.add_argument('train_dataset_type', type=str, choices=('train', 'test', 'finetune'),
                        help='train dataset type')
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('out_path', metavar='PATH', type=str, help='Output path for model')

    # Optional arguments
    parser.add_argument('--test_dataset_name', type=str, help='test dataset name')
    parser.add_argument('--test_dataset_type', type=str, choices=('train', 'test', 'finetune'),
                        help='test dataset type')
    # parser.add_argument('--train_stop_count', type=int, help='number of epochs to stop')
    parser.add_argument('--freeze_process', type=str, help='全程冻结，或者先冻结后解冻', default='part',
                        choices=('whole', 'part'))
    parser.add_argument('--samesize_reinit', type=str, default='reinit', choices=('reinit', 'keep'),
                        help='在输出类别数量相同时，是否重新初始化输出层参数')
    parser.add_argument('--save_model_count', type=int, default=5, help='saved model count during train')
    parser.add_argument('--save_model_type', type=str, choices=('accuracy', 'epoch', 'none'), default='accuracy',
                        help='model saving type during train')
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=256, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--lr-step', type=int, default=100, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.5, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=2)
    # parser.add_argument('--max_accuracy', type=float, default='0.9', help='微调时测试数据的最高准确率，针对测试数据集')
    args = parser.parse_args()
    params = vars(args)

    # torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_dataset_name = params['train_dataset_name']
    valid_datasets = datasets.__dict__.keys()
    if train_dataset_name not in valid_datasets:
        raise ValueError('Dataset {} not found. Valid arguments = {}'.format(train_dataset_name, valid_datasets))
    train_dataset_class = datasets.__dict__[train_dataset_name]
    train_dataset_modelfamily = datasets.dataset_to_modelfamily[train_dataset_name]
    train_transform = datasets.modelfamily_to_transforms[train_dataset_modelfamily]['train']
    train_dataset_type = params['train_dataset_type']
    train_dataset = train_dataset_class(train_dataset_type, transform=train_transform)

    test_dataset_name = params['test_dataset_name']
    test_dataset = None
    if test_dataset_name is not None:
        if test_dataset_name not in valid_datasets:
            raise ValueError('Dataset {} not found. Valid arguments = {}'.format(test_dataset_name, valid_datasets))
        test_dataset_class = datasets.__dict__[test_dataset_name]
        test_dataset_modelfamily = datasets.dataset_to_modelfamily[test_dataset_name]
        test_transform = datasets.modelfamily_to_transforms[test_dataset_modelfamily]['test']
        test_dataset_type = params['test_dataset_type']
        test_dataset = test_dataset_class(test_dataset_type, transform=test_transform)

    num_classes = len(train_dataset.classes)
    params['num_classes'] = num_classes

    blackbox_dir = params['victim_model_dir']
    model = Blackbox.from_modeldir(blackbox_dir, device)
    params['model_arch'] = model.model_arch
    source_model_num_classes = model.num_classes
    params['source_model_num_classes'] = source_model_num_classes
    model = model.get_model()
    freeze_process = params['freeze_process']
    samesize_reinit = params['samesize_reinit']
    if num_classes != source_model_num_classes or (
            num_classes == source_model_num_classes and samesize_reinit == 'reinit'):
        if isinstance(model, torchvision.models.AlexNet) or isinstance(model, torchvision.models.VGG):
            if freeze_process == 'part':
                for param in model.parameters():
                    param.requires_grad = False
                num_features = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(num_features, num_classes)
                model = model.to(device)
            else:
                num_features = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(num_features, num_classes)
                for param in model.parameters():
                    param.requires_grad = True
                model = model.to(device)
        elif isinstance(model, torchvision.models.ResNet) or isinstance(model,
                                                                        torchvision.models.GoogLeNet) or isinstance(
                model, torchvision.models.Inception3):
            if freeze_process == 'part':
                for param in model.parameters():
                    param.requires_grad = False
                num_ftrs = model.fc.in_features  # 获取原网络中的特征输入
                model.fc = torch.nn.Linear(num_ftrs, num_classes)
                model = model.to(device)
            else:
                num_ftrs = model.fc.in_features  # 获取原网络中的特征输入
                model.fc = torch.nn.Linear(num_ftrs, num_classes)
                for param in model.parameters():
                    param.requires_grad = True
                model = model.to(device)
        elif isinstance(model, torchvision.models.DenseNet):
            if freeze_process == 'part':
                for param in model.parameters():
                    param.requires_grad = False
                num_ftrs = model.classifier.in_features  # 获取原网络中的特征输入
                model.classifier = torch.nn.Linear(num_ftrs, num_classes)
                model = model.to(device)
            else:
                num_ftrs = model.classifier.in_features  # 获取原网络中的特征输入
                model.classifier = torch.nn.Linear(num_ftrs, num_classes)
                for param in model.parameters():
                    param.requires_grad = True
                model = model.to(device)
        else:
            print("Model Architecture not supported")
            return
    out_path = params['out_path']
    model_utils.train_model(model, train_dataset, testset=test_dataset, device=device, **params)

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
