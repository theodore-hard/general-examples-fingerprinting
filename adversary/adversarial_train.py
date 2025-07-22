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
import torch.nn as nn
import torch.optim as optim
import torchvision.models as torch_models
from torch.utils.data import DataLoader

import datasets
from utils.utils import create_dir
from victim.blackbox import Blackbox


# Define FGSM attack function
def fgsm_attack(model, loss_fn, x, y, epsilon=0.03):
    """
    Generate adversarial examples using FGSM
    Args:
        model: Target model to attack
        loss_fn: Loss function
        x: Clean input tensor (batch_size, 3, 32, 32)
        y: True labels
        epsilon: Perturbation magnitude(cifar10 cifar 100 0.03   imagenet 0.007)
    Returns:
        Adversarial examples tensor
    """
    x.requires_grad = True
    if (isinstance(model, torch_models.GoogLeNet)):
        outputs, aux2, aux1 = model(x)
    else:
        outputs = model(x)
    # outputs = model(x)
    loss = loss_fn(outputs, y)
    model.zero_grad()
    loss.backward()

    # Generate adversarial examples
    grad_sign = x.grad.data.sign()
    x_adv = x + epsilon * grad_sign
    # return torch.clamp(x_adv, 0, 1).detach()
    return x_adv.detach()


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('dataset_name', type=str, help=' dataset name')
    parser.add_argument('model_dir', type=str,
                        help='Path to model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('out_path', metavar='PATH', type=str, help='Output path for model')

    # Optional arguments
    # parser.add_argument('--transform', type=str, help='specify a designated transform')
    parser.add_argument('--epsilon', type=float, default=0.03, help='Perturbation magnitude')
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=2)
    args = parser.parse_args()
    params = vars(args)

    # torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up dataset
    dataset_name = params['dataset_name']
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset_class = datasets.__dict__[dataset_name]
    dataset_modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    train_transform = datasets.modelfamily_to_transforms[dataset_modelfamily]['train']
    test_transform = datasets.modelfamily_to_transforms[dataset_modelfamily]['test']
    # transform = params['transform']
    # if transform is not None:
    #     transform = datasets.modelfamily_to_transforms[transform]['test']
    train_dataset = dataset_class('train', transform=train_transform)
    test_dataset = dataset_class('test', transform=test_transform)
    batch_size = params['batch_size']
    num_workers = params['num_workers']
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                 pin_memory=True)

    blackbox_dir = params['model_dir']
    blackbox = Blackbox.from_modeldir(blackbox_dir, device)
    model = blackbox.get_model()
    num_classes = len(train_dataset.classes)
    params['num_classes'] = num_classes

    # ----------- Train
    out_path = params['out_path']
    lr = params['lr']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create directory for saving models
    # os.makedirs('saved_models', exist_ok=True)

    # Adversarial training parameters
    total_epochs = 270
    save_interval = 30
    # epsilon = 0.03
    adv_batch_size = 128
    # adv_batch_size = 64
    epsilon = params['epsilon']
    # Training loop
    # for epoch in range(total_epochs):
    #    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        model.train()
        # Generate adversarial batch
        adv_inputs = fgsm_attack(model, criterion, inputs[:adv_batch_size],
                                 labels[:adv_batch_size], epsilon)

        # Combine clean and adversarial data
        # combined_inputs = torch.cat([inputs, adv_inputs], dim=0)
        # combined_labels = torch.cat([labels, labels[:adv_batch_size]], dim=0)

        # Forward pass
        optimizer.zero_grad()
        if (isinstance(model, torch_models.GoogLeNet)):
            outputs, aux2, aux1 = model(adv_inputs)
        else:
            outputs = model(adv_inputs)
        # outputs = model(adv_inputs)
        loss = criterion(outputs, labels)
        # outputs = model(combined_inputs)
        # loss = criterion(outputs, combined_labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # running_loss += loss.item()

        # print(f'iter [{i + 1}/{total_epochs}], Loss: {running_loss / len(train_dataloader):.4f}')
        print(f'iter [{i + 1}/{total_epochs}], Loss: {loss}')
        # Save model periodically
        if (i + 1) % save_interval == 0:
            # torch.save(model.state_dict(), f'saved_models/model_epoch_{epoch + 1}.pth')
            # print(f'Model saved at epoch {epoch + 1}')
            iter = i + 1
            model_out_dir = osp.join(out_path, str(iter))
            create_dir(model_out_dir)
            model_out_path = osp.join(model_out_dir, 'checkpoint{}.pth.tar'.format(''))

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_acc = 100 * correct / total
            state = {
                'epoch': i,
                'arch': model.__class__,
                'state_dict': model.state_dict(),
                'best_acc': test_acc,
                'optimizer': optimizer.state_dict(),
                'created_on': str(datetime.now()),
            }
            torch.save(state, model_out_path)
            params['best_acc'] = test_acc
            params['model_arch'] = blackbox.model_arch
            params['num_classes'] = blackbox.num_classes
            params['created_on'] = str(datetime.now())
            params_out_path = osp.join(model_out_dir, 'params.json')
            with open(params_out_path, 'w') as jf:
                json.dump(params, jf, indent=True)
            print(f'Model saved at epoch {i + 1}, test_acc={test_acc}')
        if (i + 1) >= total_epochs:
            break


if __name__ == '__main__':
    main()
