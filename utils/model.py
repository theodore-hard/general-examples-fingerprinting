#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import os
import os.path as osp
import time
from collections import defaultdict as dd
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as torch_models
from torch.utils.data import DataLoader

def create_dir(dir_path):
    if not osp.exists(dir_path):
        print('Path {} does not exist. Creating it...'.format(dir_path))
        os.makedirs(dir_path)

def clear_folder(path):
    # 遍历所有文件和文件夹
    for file_name in os.listdir(path):
        # 如果是文件则删除
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        # 如果是文件夹则递归删除
        elif os.path.isdir(file_path):
            clear_folder(file_path)
            os.rmdir(file_path)
    os.rmdir(path)


output_buffer = []
output_buffer_path = []
def save_model(out_path, model, test_acc, epoch, save_model_count=5):
    global  output_buffer
    if len(output_buffer) < save_model_count:
        model_out_path = osp.join(out_path,str(test_acc)+"_"+str(epoch))
        if not os.path.exists(model_out_path):
            os.mkdir(model_out_path)
            print("create directory", model_out_path)
        model_out_path_file = osp.join(model_out_path, 'checkpoint.pth.tar')
        torch.save(model, model_out_path_file)
        output_buffer.append(test_acc)
        output_buffer_path.append(model_out_path)
    else:
        min_value = min(output_buffer)
        if min_value <= test_acc:
            min_position = [i for i, x in enumerate(output_buffer) if x == min(output_buffer)]
            replace_position = min_position[0]
            model_out_path = osp.join(out_path, str(test_acc) + "_" + str(epoch))
            if not os.path.exists(model_out_path):
                os.mkdir(model_out_path)
                print("create directory", model_out_path)
            model_out_path_file = osp.join(model_out_path, 'checkpoint.pth.tar')
            clear_folder(output_buffer_path[replace_position])
            torch.save(model, model_out_path_file)
            output_buffer[replace_position] = test_acc
            output_buffer_path[replace_position] =model_out_path


def get_net(model_name, n_output_classes=1000, **kwargs):
    print('=> loading model {} with arguments: {}'.format(model_name, kwargs))
    valid_models = [x for x in torch_models.__dict__.keys() if not x.startswith('__')]
    if model_name not in valid_models:
        raise ValueError('Model not found. Valid arguments = {}...'.format(valid_models))
    model = torch_models.__dict__[model_name](**kwargs)
    # Edit last FC layer to include n_output_classes
    if n_output_classes != 1000:
        if 'squeeze' in model_name:
            model.num_classes = n_output_classes
            model.classifier[1] = nn.Conv2d(512, n_output_classes, kernel_size=(1, 1))
        elif 'alexnet' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_output_classes)
        elif 'vgg' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_output_classes)
        elif 'dense' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, n_output_classes)
        else:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, n_output_classes)
    return model


def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))


def train_step(model, train_loader, criterion, optimizer, epoch, device, log_interval=10, writer=None):
    model.train()
    train_loss = 0.
    correct = 0
    topk_running_corrects = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if(isinstance(model, torch_models.GoogLeNet)):
            outputs,aux2,aux1 = model(inputs)
        else:
            outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if writer is not None:
            pass

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        _, topk_preds = torch.topk(outputs, 5)
        total += targets.size(0)
        if len(targets.size()) == 2:
            # Labels could be a posterior probability distribution. Use argmax as a proxy.
            target_probs, target_labels = targets.max(1)
        else:
            target_labels = targets

        correct += predicted.eq(target_labels).sum().item()
        for index, topk_pred in enumerate(topk_preds):
            if target_labels.data[index] in topk_pred:
                topk_running_corrects += 1

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        acc = 100. * correct / total
        train_loss_batch = train_loss / total


        if writer is not None:
            writer.add_scalar('Loss/train', loss.item(), exact_epoch)
            writer.add_scalar('Accuracy/train', acc, exact_epoch)

    t_end = time.time()
    acc = 100. * correct / total

    return train_loss_batch, acc


def test_step(model, test_loader, criterion, device, epoch=0., silent=False, writer=None):
    model.eval()
    test_loss = 0.
    correct = 0
    topk_running_corrects = 0
    total = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            _, topk_preds = torch.topk(outputs, 5)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            for index, topk_pred in enumerate(topk_preds):
                if targets.data[index] in topk_pred:
                    topk_running_corrects += 1

    t_end = time.time()
    t_epoch = int(t_end - t_start)

    acc = 100. * correct / total
    topk_acc = 100. * topk_running_corrects / total
    test_loss /= total

    if not silent:
        print('[Test]  Epoch: {}\tLoss: {:.6f}\tAcc: {:.1f}% ({}/{})\ttopk_Acc: {:.1f}% ({}/{})'.format(epoch, test_loss, acc,
                                                                             correct, total, topk_acc, topk_running_corrects, total))

    if writer is not None:
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)

    return test_loss, acc


def train_model(model, trainset, out_path, batch_size=64, criterion_train=None, criterion_test=None, testset=None,
                device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                epochs=100, log_interval=100, save_model_count=5, save_model_type='accuracy', weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
                 writer=None, **kwargs):
    if device is None:
        device = torch.device('cuda')
    if not osp.exists(out_path):
        create_dir(out_path)
    run_id = str(datetime.now())

    # Data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        test_loader = None

    if weighted_loss:
        if not isinstance(trainset.samples[0][1], int):
            print('Labels in trainset is of type: {}. Expected: {}.'.format(type(trainset.samples[0][1]), int))

        class_to_count = dd(int)
        for _, y in trainset.samples:
            class_to_count[y] += 1
        class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
        print('=> counts per class: ', class_sample_count)
        weight = np.min(class_sample_count) / torch.Tensor(class_sample_count)
        weight = weight.to(device)
        print('=> using weights: ', weight)
    else:
        weight = None

    # Optimizer
    if criterion_train is None:
        criterion_train = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if criterion_test is None:
        criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    best_train_acc, train_acc = -1., -1.
    best_test_acc, last_best_acc, test_acc, test_loss = -1., -1., -1., -1.

    # Resume if required
    if resume is not None:
        model_path = resume
        if osp.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            best_test_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging
    log_path = osp.join(out_path, 'train{}.log.tsv'.format(checkpoint_suffix))
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy']
            wf.write('\t'.join(columns) + '\n')

    model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
    #train_stop_count = 5 #如果连续两次epoch，训练准确度都没有提升，则停止训练
    train_desc_times = 0
    stop_epoch = epochs + 1
    for epoch in range(start_epoch, stop_epoch):
        train_loss, train_acc = train_step(model, train_loader, criterion_train, optimizer, epoch, device,
                                           log_interval=log_interval)
        scheduler.step(epoch)
        print('[Train] Epoch: {:.2f} \ttrain_loss: {:.6f}\ttrain_acc: {:.1f}'.format(epoch, train_loss, train_acc))
        best_train_acc = max(best_train_acc, train_acc)


        if test_loader is not None:
            test_loss, test_acc = test_step(model, test_loader, criterion_test, device, epoch=epoch)
            best_test_acc = max(best_test_acc, test_acc)

        # Checkpoint
        if test_acc >= best_test_acc:
            state = {
                'epoch': epoch,
                'arch': model.__class__,
                'state_dict': model.state_dict(),
                'best_acc': test_acc,
                'optimizer': optimizer.state_dict(),
                'created_on': str(datetime.now()),
            }
            torch.save(state, model_out_path)
            if save_model_type == 'accuracy' and test_acc > 0 and test_acc > last_best_acc:
                save_model(out_path, state, test_acc, epoch, save_model_count=save_model_count)
                last_best_acc = test_acc
            elif save_model_type== 'epoch':
                save_model(out_path, state, test_acc, epoch, save_model_count=save_model_count)


        # Log
        with open(log_path, 'a') as af:
            train_cols = [run_id, epoch, 'train', train_loss, train_acc, best_train_acc]
            af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            test_cols = [run_id, epoch, 'test', test_loss, test_acc, best_test_acc]
            af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    return model


def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    topk_running_corrects = 0
    total = 0
    t_start = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            _, topk_preds = torch.topk(outputs, 5)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            for index, topk_pred in enumerate(topk_preds):
                if targets.data[index] in topk_pred:
                    topk_running_corrects += 1
    t_end = time.time()
    t_epoch = int(t_end - t_start)
    acc = 100. * correct / total
    topk_acc = 100. * topk_running_corrects / total
    print('Time: {}\tAcc: {:.1f}% ({}/{})\ttopk_Acc: {:.1f}% ({}/{})'.format(t_epoch, acc, correct, total, topk_acc, topk_running_corrects, total))
    return acc, topk_acc

def test_half_model(model, test_loader, device):
    model.eval()
    correct = 0
    topk_running_corrects = 0
    total = 0
    t_start = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.half().to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            _, topk_preds = torch.topk(outputs, 5)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            for index, topk_pred in enumerate(topk_preds):
                if targets.data[index] in topk_pred:
                    topk_running_corrects += 1
    t_end = time.time()
    t_epoch = int(t_end - t_start)
    acc = 100. * correct / total
    topk_acc = 100. * topk_running_corrects / total
    print('Time: {}\tAcc: {:.1f}% ({}/{})\ttopk_Acc: {:.1f}% ({}/{})'.format(t_epoch, acc, correct, total, topk_acc, topk_running_corrects, total))
    return acc, topk_acc