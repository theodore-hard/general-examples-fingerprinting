#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import math
import os
import os.path as osp

import torch
import torch.nn.functional as F


from victim.blackbox import Blackbox

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


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
    dest_image_path = osp.join(out_path, 'data.pt')
    fingerprints_data = torch.load(dest_image_path)
    print("-> verify ownership...")
    test_x, test_y = fingerprints_data["test_x"], fingerprints_data["test_y"]
    y1, y2 = test_y.cpu(), []
    with torch.no_grad():
        y2.append(batch_forward(model=model, x=test_x, argmax=True))
    y2 = torch.cat(y2)
    matching_rate = round(float(y1.eq(y2.view_as(y1)).sum()) / len(y1), 5)
    print(f"matching_rate:{matching_rate}")


if __name__ == "__main__":
    main()
