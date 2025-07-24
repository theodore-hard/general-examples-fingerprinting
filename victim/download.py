import argparse
import json
import os
import os.path as osp
from datetime import datetime

import torch
from torch.utils.data import DataLoader

import models.zoo as zoo
import utils.model as model_utils
import datasets


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('dataset', metavar='DS_NAME', type=str, help='Dataset name')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    # Optional arguments
    parser.add_argument('-o', '--out_path', metavar='PATH', type=str, help='Output path for model')
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=2)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)

    args = parser.parse_args()
    params = vars(args)

    # torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up dataset
    dataset_name = params['dataset']
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]
    model_name = params['model_arch']
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    testset = dataset('test', transform=test_transform)
    num_classes = len(testset.classes)
    params['num_classes'] = num_classes


    pretrained = params['pretrained']
    if pretrained is not None:
        model = zoo.get_net(model_name, num_classes=num_classes, pretrained=True)
    else:
        model = zoo.get_net(model_name, num_classes=num_classes, pretrained=False)  #model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    model.eval()

    out_path = params['out_path']
    batch_size = params['batch_size']
    num_workers = params['num_workers']
    model_out_path = osp.join(out_path, 'checkpoint.pth.tar')
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_acc, topk_acc = model_utils.test_model(model, test_loader, device=device)
    state = {
        'epoch': 0,
        'arch': model.__class__,
        'state_dict': model.state_dict(),
        'best_acc': test_acc,
        'optimizer': None,
        'created_on': str(datetime.now()),
        'topk_acc':topk_acc
    }
    torch.save(state, model_out_path)

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()