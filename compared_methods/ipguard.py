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
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchattacks.attack import Attack
from tqdm import tqdm

from fingerprint import datasets
from fingerprint.victim.blackbox import Blackbox

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class Adv(Attack):
    def __init__(self, model):
        super().__init__("Adv", model)
        self._supported_mode = ['default', 'targeted']

    def IPGuard(self, images, labels, k, targeted, steps=1000, lr=1e-2):
        assert targeted in ["L", "R"]
        batch_x = images.clone().detach()

        # We find k should be very small, e.g., k=0.01, since logit is always small than 10
        k = Variable(torch.Tensor([k]), requires_grad=True)[0].detach()

        adv_x = []
        self.model.eval()
        batch_size = len(labels)
        phar = tqdm(range(batch_size))
        ReLU = torch.nn.ReLU()

        for idx in phar:
            x = batch_x[[idx]].clone().to(self.device)
            z = self.model(x)
            i = z.argmax(dim=1)[0]
            if targeted == "L":  # least-like
                j = z.argmin(dim=1)[0]
            else:  # random
                ll = list(range(z.shape[1]))
                ll.remove(int(i))
                j = random.choice(ll)

            for step in range(steps):
                x = x.detach()
                x.requires_grad = True
                optimizer = torch.optim.Adam([x], lr=lr)
                optimizer.zero_grad()

                if z.shape[1] > 2:
                    z = self.model(x)
                    z[0][i] = -1000
                    z[0][j] = -1000
                    t = z.argmax(dim=1)[0]
                    z = self.model(x)
                    loss = ReLU(z[0][i] - z[0][j] + k) + ReLU(z[0][t] - z[0][i])
                else:
                    # compatible binary classifier
                    z = self.model(x)
                    loss = ReLU(z[0][i] - z[0][j] + k)
                    t = j
                loss.backward()
                optimizer.step()
                phar.set_description(
                    f"-> [IPGuard] idx{idx}-step{step} i:{int(i)} j:{int(j)} t:{int(t)} "
                    f"z_i:{round(float(self.model(x)[0][i]), 4)} z_j:{round(float(self.model(x)[0][j]), 4)} loss:{round(float(loss.data), 4)}")

                # loss ≈ 0
                if loss <= 1e-5:
                    break

            z = self.model(x)[0]
            print(f"-> max_logit={round(float(torch.max(z)), 5)}, "
                  f"z_j={round(float(z[j]), 4)} ≥ z_i={round(float(z[i]), 4)} + {k} \n")
            adv_x.append(x.detach().cpu())

        batch_x = torch.cat(adv_x).to(self.device)
        batch_y = batch_forward(self.model, batch_x, batch_size=200, argmax=True)
        return batch_x.cpu().detach(), batch_y.cpu().detach()


def set_default_seed(seed=1000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


class IPGuard():
    def __init__(self, model, test_loader, out_root, device, k, targeted="L",
                 steps=1000, max_count=200, seed=100):
        self.k = k
        self.steps = steps
        self.test_size = max_count
        self.targeted = targeted

        # init logger
        self.logger = logging.getLogger('IPGuard')

        # init dataset
        self.seed = seed
        # self.dataset = model.dataset_id
        self.test_loader = test_loader
        # self.bounds = self.test_loader.bounds

        # init model
        # self.task = model.task
        self.device = device
        self.model = model.to(self.device)
        self.out_root = out_root

    def extract(self):
        """
        extract fingerprint samples.
        :return:
        """
        self.logger.info("-> extract fingerprint...")
        adv = Adv(self.model)
        count = 0
        test_x, test_y = [], []
        for x, y in self.test_loader:
            x, y = adv.IPGuard(x, y, k=self.k, targeted=self.targeted, steps=self.steps)
            test_x.append(x)
            test_y.append(y)
            count += len(x)
            if count >= self.test_size:
                break

        fingerprint = {
            "test_x": torch.cat(test_x)[:self.test_size],
            "test_y": torch.cat(test_y)[:self.test_size]
        }
        dest_image_path = osp.join(self.out_root, 'data.pt')
        torch.save(fingerprint, dest_image_path)
        return fingerprint

    def verify(self, fingerprint, model):
        """
        verify ownership between model1 & model2
        :return:
        """
        self.logger.info("-> verify ownership...")
        test_x, test_y = fingerprint["test_x"], fingerprint["test_y"]
        y1, y2 = test_y.cpu(), []
        with torch.no_grad():
            y2.append(batch_forward(model=model, x=test_x, argmax=True))
        y2 = torch.cat(y2)
        matching_rate = round(float(y1.eq(y2.view_as(y1)).sum()) / len(y1), 5)
        self.logger.info(f"matching_rate:{matching_rate}")
        return {"MR": matching_rate}

    def compare(self, model):
        return self.verify(self.extract(), model)


"""

"""


def main():
    parser = argparse.ArgumentParser(description="Build micro benchmark.")
    parser.add_argument("dataset_name", type=str, help="extract dataset name")
    parser.add_argument("dataset_type", type=str, help="extract dataset type")
    parser.add_argument('victim_model_dir', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('out_path', type=str, help='Output path for fingerprints')
    parser.add_argument('max_count', type=int, help='Output image max count')

    parser.add_argument("--k", default=0.01, type=float, help="k of IPGuard, cifar-10(10), cifar-100(10),imagenet(100)")
    parser.add_argument("-targeted", default="L", type=str, help="L:lest-likely R:random", choices=["L", "R"])
    parser.add_argument('--device_id', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument("-seed", default=1000, type=int, help="Default seed of numpy/pyTorch")
    parser.add_argument('--steps', type=int, default=1000, help='iteration steps')
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
    out_path = params['out_path']
    max_count = params['max_count']
    set_default_seed(args.seed)

    k = params['k']
    since = time.time()
    steps = params['steps']
    ipguard = IPGuard(model=model, test_loader=dataloader, device=device, out_root=out_path, max_count=max_count,
                      targeted=args.targeted, k=k, seed=args.seed, steps=steps)
    ipguard.extract()
    total_time_elapsed = time.time() - since
    calc_time = f'{total_time_elapsed // 60:.0f}m {total_time_elapsed % 60:.0f}s'
    params['created_on'] = str(datetime.now())
    params['calc_time'] = calc_time
    params_out_path = osp.join(out_path, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == "__main__":
    main()
