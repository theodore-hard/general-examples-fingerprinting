#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os

import torch
from torch.quantization import quantize_dynamic
from torch.utils.data import DataLoader

import utils.model as model_utils
import datasets
from victim.blackbox import Blackbox


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('test_dataset_name', type=str, help='test dataset name')
    parser.add_argument('test_dataset_type', type=str, choices=('train', 'test', 'finetune'), help='test dataset type')
    parser.add_argument('model_dir', type=str, help='Path to model. Should contain files "model_best.pth.tar" and "params.json"')
    # Optional arguments
    parser.add_argument('--transform', type=str, help='specify a designated transform')
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=2)
    parser.add_argument("--quantization", default="no", type=str, help="int8 half no", choices=["int8","half","no"])
    args = parser.parse_args()
    params = vars(args)

    # torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up dataset
    test_dataset_name = params['test_dataset_name']
    valid_datasets = datasets.__dict__.keys()
    if test_dataset_name not in valid_datasets:
        raise ValueError('Dataset {} not found. Valid arguments = {}'.format(test_dataset_name, valid_datasets))
    test_dataset_class = datasets.__dict__[test_dataset_name]
    train_dataset_modelfamily = datasets.dataset_to_modelfamily[test_dataset_name]
    test_transform = datasets.modelfamily_to_transforms[train_dataset_modelfamily]['test']
    transform = params['transform']
    if transform is not None:
        test_transform = datasets.modelfamily_to_transforms[transform]['test']
    batch_size = params['batch_size']
    num_workers = params['num_workers']
    test_dataset_type = params['test_dataset_type']
    testset = test_dataset_class(test_dataset_type, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    blackbox_dir = params['model_dir']
    quantization = params['quantization']
    if quantization == 'no':
        model = Blackbox.from_modeldir(blackbox_dir, device).get_model()
        model_utils.test_model(model,  test_loader, device=device)
    elif quantization == 'int8':
        model = Blackbox.from_modeldir(blackbox_dir, device).get_model()
        quantized_model = quantize_dynamic(model,
            {torch.nn.Conv2d, torch.nn.Linear},
            dtype=torch.qint8
        )
        model_utils.test_model(quantized_model, test_loader, device=device)


if __name__ == '__main__':
    main()
