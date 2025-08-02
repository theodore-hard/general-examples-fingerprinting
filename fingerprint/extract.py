#!/usr/bin/env python
# -*- coding:utf-8 -*-


import argparse
import datetime
import json
import logging
import math
import os
import os.path as osp
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import SSIM
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchattacks.attack import Attack
from torchvision.transforms import transforms
from tqdm import tqdm

import datasets
from victim.blackbox import Blackbox

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def create_dir(dir_path):
    if not osp.exists(dir_path):
        print('Path {} does not exist. Creating it...'.format(dir_path))
        os.makedirs(dir_path)

def atanh(x, eps=1e-6):
    """
    The inverse hyperbolic tangent function, missing in pytorch.

    :param x: a tensor or a Variable
    :param eps: used to enhance numeric stability
    :return: :math:`\\tanh^{-1}{x}`, of the same type as ``x``
    """
    x = x * (1 - eps)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))


def to_tanh_space(x, box):
    # type: (Union[Variable, torch.FloatTensor], Tuple[float, float]) -> Union[Variable, torch.FloatTensor]
    """
    Convert a batch of tensors to tanh-space. This method complements the
    implementation of the change-of-variable trick in terms of tanh.

    :param x: the batch of tensors, of dimension [B x C x H x W]
    :param box: a tuple of lower bound and upper bound of the box constraint
    :return: the batch of tensors in tanh-space, of the same dimension;
             the returned tensor is on the same device as ``x``
    """
    _box_mul = (box[1] - box[0]) * 0.5
    _box_plus = (box[1] + box[0]) * 0.5
    return atanh((x - _box_plus) / _box_mul)


def from_tanh_space(x, box):
    # type: (Union[Variable, torch.FloatTensor], Tuple[float, float]) -> Union[Variable, torch.FloatTensor]
    """
    Convert a batch of tensors from tanh-space to oridinary image space.
    This method complements the implementation of the change-of-variable trick
    in terms of tanh.

    :param x: the batch of tensors, of dimension [B x C x H x W]
    :param box: a tuple of lower bound and upper bound of the box constraint
    :return: the batch of tensors in ordinary image space, of the same
             dimension; the returned tensor is on the same device as ``x``
    """
    _box_mul = (box[1] - box[0]) * 0.5
    _box_plus = (box[1] + box[0]) * 0.5
    return torch.tanh(x) * _box_mul + _box_plus


def extract_general_examples(model, images, labels, precision, targeted, steps=1000, lr=1e-2, box=(-1., 1.), device=-1):
    assert targeted in ["L", "R"]
    batch_x = images.clone().detach()
    stop_threshold = 1 - precision

    adv_x = []
    model.eval()
    batch_size = len(labels)
    phar = tqdm(range(batch_size))
    ReLU = torch.nn.ReLU()
    for idx in phar:
        x = batch_x[[idx]].clone().to(device)
        source_output = model(x)
        i = source_output.argmax(dim=1)[0]
        l = source_output.argmin(dim=1)[0]
        span = source_output[0][i] - source_output[0][l]
        span = span.cpu().item()

        # Compute the absolute distance between each element and the maximum value per row
        max_values, max_indices = torch.max(source_output, dim=1, keepdim=True)
        distances = torch.abs(source_output - max_values)
        # Set distance at maximum value positions to 0 (to be excluded later)
        distances.scatter_(1, max_indices, 0)
        # Calculate average of non-zero distances (excluding the maximum value itself)
        avg_distances = distances.sum(dim=1) / (source_output.size(1) - 1)
        max_value = max_values[0].cpu().item()
        avg_distance = avg_distances[0].cpu().item()

        print(f"i={i},l={l},span={span},max_value={max_value},avg_distance={avg_distance}")

        source_softmax = F.softmax(source_output, dim=1)
        source_prob = source_softmax[0][i].item()
        print(f"source_prob={source_prob}>precision={precision}")
        if source_prob < precision:
            continue
        if targeted == "L":  # least-like
            j = source_output.argmin(dim=1)[0]
        else:  # random
            ll = list(range(source_output.shape[1]))
            ll.remove(int(i))
            j = random.choice(ll)

        source_label_value = source_output[0][i].cpu().item()
        pert_tanh = torch.zeros(x.size()).to(device)
        pert_tanh_var = Variable(pert_tanh, requires_grad=True)

        x = x.detach()
        x.requires_grad = False
        inputs_tanh = to_tanh_space(x, box).to(device)
        inputs_tanh_var = Variable(inputs_tanh, requires_grad=False)

        optimizer = torch.optim.Adam([pert_tanh_var], lr=lr)
        for step in range(steps):
            advxs_var = from_tanh_space(inputs_tanh_var + pert_tanh_var, box)

            z = model(advxs_var)
            z[0][j] = -1000
            t_max = z.argmax(dim=1)[0]
            z[0][j] = 1000
            t_min = z.argmin(dim=1)[0]

            linear_output = model(advxs_var)
            softmax_outputs = F.softmax(linear_output, dim=1)
            pred_loss = ReLU(1.0 - softmax_outputs[0][j])
            pred_value = pred_loss.cpu().item()

            target_value = linear_output[0][j]
            second_largest_value = linear_output[0][t_max].cpu().item()
            loss = ReLU(second_largest_value - target_value + span)

            if pred_value <= stop_threshold:
                print(
                    f"-> label_prob={softmax_outputs[0][j]} ≥  {precision}  target_value={linear_output[0][j]} ≥ {source_label_value}\n")
                adv_x.append(x.detach().cpu())
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            phar.set_description(f"->idx{idx}-step{step} i:{int(i)} j:{int(j)} loss:{round(float(loss.data), 4)}")

    if len(adv_x) > 0:
        batch_x = torch.cat(adv_x).to(device)
        batch_y = batch_forward(model, batch_x, batch_size=200, argmax=True)
        return batch_x.cpu().detach(), batch_y.cpu().detach()
    else:
        return None, None


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
            pred.append(model(batch_x).cpu().detach())
    pred = torch.cat(pred)
    return pred.argmax(dim=1) if argmax else pred


def batch_mid_forward(model, x, layer_index, batch_size=200):
    steps = math.ceil(len(x) / batch_size)
    device = next(model.parameters()).device
    outputs = []
    with torch.no_grad():
        for step in range(steps):
            off = (step * batch_size)
            batch_x = x[off: off + batch_size].clone().to(device)
            batch_out = model.mid_forward(batch_x, layer_index=layer_index).detach().cpu()
            outputs.append(batch_out)
        del batch_x, batch_out, x
        outputs = torch.cat(outputs).detach().cpu()
        torch.cuda.empty_cache()
    return outputs


# Obtain the maximum variance
def get_max_min(transform_train):
    norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
    mean_list = norm_transform[0].mean
    std_list = norm_transform[0].std
    min_list = [0.0, 0.0, 0.0]
    min_list = [(min_list[i] - mean_list[i]) / std_list[i] for i in range(len(mean_list))]
    min_value = min(min_list)
    max_list = [1.0, 1.0, 1.0]
    max_list = [(max_list[i] - mean_list[i]) / std_list[i] for i in range(len(mean_list))]
    max_value = max(max_list)
    print(f'min_value:{min_value}, max_value:{max_value}')
    return min_value, max_value


def generate_fingerprints(model, test_loader, transform, out_root, precision, targeted="L", steps=100, lr=0.01,
                          max_count=200, device=-1):
    """
    extract fingerprint samples.
    :return:
    """
    print("-> extract fingerprint...")
    count = 0
    test_x, test_y = [], []
    min, max = get_max_min(transform)
    for x, y in test_loader:
        since = time.time()
        x, y = extract_general_examples(model, x, y, precision=precision, targeted=targeted, steps=steps,
                                        box=(min, max), lr=lr, device=device)
        if x is not None:
            test_x.append(x)
            test_y.append(y)
            count += len(x)
            total_time_elapsed = time.time() - since
            print(f"result_count={count} time_generate={total_time_elapsed}s")
            if count >= max_count:
                break

    fingerprint = {
        "test_x": torch.cat(test_x)[:max_count],
        "test_y": torch.cat(test_y)[:max_count]
    }
    dest_image_path = osp.join(out_root, 'data.pt')
    torch.save(fingerprint, dest_image_path)
    return fingerprint


def main():
    parser = argparse.ArgumentParser(description="Build micro benchmark.")
    parser.add_argument("dataset_name", type=str, help="extract dataset name")
    parser.add_argument("dataset_type", type=str, help="extract dataset type")
    parser.add_argument('victim_model_dir', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('out_path', type=str, help='Output path for fingerprints')
    parser.add_argument('max_count', type=int, help='Output image max count')

    parser.add_argument("--precision", default=0.999, type=float, help="precision for target label")
    parser.add_argument("--targeted", default="L", type=str, help="L:lest-likely R:random", choices=["L", "R"])
    parser.add_argument('--device_id', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--steps', type=int, default=100, help='iteration steps')
    args, unknown = parser.parse_known_args()
    args.ROOT = ROOT
    args = parser.parse_args()
    params = vars(args)

    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset_name = params['dataset_name']
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset {} not found. Valid arguments = {}'.format(dataset_name, valid_datasets))
    dataset_class = datasets.__dict__[dataset_name]
    dataset_modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    transform = datasets.modelfamily_to_transforms[dataset_modelfamily]['test']
    dataset_type = params['dataset_type']
    dataset = dataset_class(dataset_type, transform=transform)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    blackbox_dir = params['victim_model_dir']
    model = Blackbox.from_modeldir(blackbox_dir, device).get_model()
    for p in model.parameters():
        p.requires_grad = False
    out_path = params['out_path']
    create_dir(out_path)

    max_count = params['max_count']

    precision = params['precision']
    since = time.time()
    lr = params['lr']
    steps = params['steps']
    targeted = params['targeted']
    generate_fingerprints(model=model, test_loader=dataloader, transform=transform, device=device, out_root=out_path,
                          max_count=max_count, targeted=targeted, precision=precision, lr=lr, steps=steps)
    total_time_elapsed = time.time() - since
    calc_time = f'{total_time_elapsed // 60:.0f}m {total_time_elapsed % 60:.0f}s'
    params['created_on'] = str(datetime.now())
    params['calc_time'] = calc_time
    params_out_path = osp.join(out_path, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == "__main__":
    main()
