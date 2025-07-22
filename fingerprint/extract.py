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


class Adv(Attack):
    def __init__(self, model):
        super().__init__("Adv", model)
        self._supported_mode = ['default', 'targeted']

    def General(self, images, labels, precision, targeted, generated_image, steps=1000, lr=1e-2, box=(-1., 1.)):
        assert targeted in ["L", "R"]
        assert generated_image in ["S", "R"]
        batch_x = images.clone().detach()
        stop_threshold = 1 - precision

        adv_x = []
        self.model.eval()
        batch_size = len(labels)
        phar = tqdm(range(batch_size))
        ReLU = torch.nn.ReLU()
        for idx in phar:
            x = batch_x[[idx]].clone().to(self.device)
            source_output = self.model(x)
            i = source_output.argmax(dim=1)[0]
            l = source_output.argmin(dim=1)[0]
            span = source_output[0][i] - source_output[0][l]
            span = span.cpu().item()

            # 计算span平均值
            # 计算每行元素与最大值的绝对距离
            max_values, max_indices = torch.max(source_output, dim=1, keepdim=True)
            distances = torch.abs(source_output - max_values)
            # 将最大值位置的距离设为0（后续排除）
            distances.scatter_(1, max_indices, 0)
            # 计算非零距离的平均值（排除最大值自身）
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
            pert_tanh = torch.zeros(x.size()).to(self.device)
            pert_tanh_var = Variable(pert_tanh, requires_grad=True)

            if generated_image == "R":
                # Create random tensor with same shape (values 0-1 by default)
                random_tensor = torch.rand_like(x).clone().to(self.device)
                # If reference has negative values (normalized), adjust range to [-1, 1]
                if x.min() < 0:
                    x = random_tensor * 2 - 1
                else:
                    x = random_tensor
            x = x.detach()
            x.requires_grad = False

            inputs_tanh = to_tanh_space(x, box).to(self.device)
            inputs_tanh_var = Variable(inputs_tanh, requires_grad=False)

            optimizer = torch.optim.Adam([pert_tanh_var], lr=lr)
            for step in range(steps):
                advxs_var = from_tanh_space(inputs_tanh_var + pert_tanh_var, box)

                z = self.model(advxs_var)
                z[0][j] = -1000
                t_max = z.argmax(dim=1)[0]
                z[0][j] = 1000
                t_min = z.argmin(dim=1)[0]

                linear_output = self.model(advxs_var)
                softmax_outputs = F.softmax(linear_output, dim=1)
                # pred_loss = ReLU(precision - softmax_outputs[0][j])
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
                phar.set_description(
                    f"->idx{idx}-step{step} i:{int(i)} j:{int(j)} loss:{round(float(loss.data), 4)}")

        if len(adv_x) > 0:
            batch_x = torch.cat(adv_x).to(self.device)
            batch_y = batch_forward(self.model, batch_x, batch_size=200, argmax=True)
            return batch_x.cpu().detach(), batch_y.cpu().detach()
        else:
            return None, None

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


class General():
    def __init__(self, model, test_loader, transform, out_root, device, precision, targeted="L",
                 steps=100, lr=0.01, max_count=200, seed=100, alpha=1.0, beta=0.00001, generated_image="S"):
        self.precision = precision
        self.steps = steps
        self.test_size = max_count
        self.targeted = targeted

        # init logger
        self.logger = logging.getLogger('IPGuard')

        # init dataset
        # self.seed = seed
        # self.dataset = model.dataset_id
        self.test_loader = test_loader
        # self.bounds = self.test_loader.bounds

        # init model
        # self.task = model.task
        self.device = device
        self.model = model.to(self.device)
        self.out_root = out_root
        self.transform = transform
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.generated_image = generated_image

    # 获取最大方差，用以计算tensor转换后，最小的l2范数距离
    def get_max_min(self, transform_train):
        # 取出标准化的 transform
        # filter用法：用于过滤序列，过滤掉不符合条件的元素，返回符合条件的元素组成新列表。
        # filter(function,iterable)，function -- 判断函数，iterable -- 可迭代对象
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        # 取出均值
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

    def tensor_to_random_image(self, image_tensor, save_dir=None):
        """
         Generate random images matching dimensions of transformed tensor

         Args:
             tensor (torch.Tensor): Transformed tensor (NCHW format expected)
             save_dir (str, optional): Directory to save generated images

         Returns:
             list: List of PIL.Image objects (single image if batch=1)
         """
        # Ensure tensor is on CPU and convert to numpy
        if image_tensor.device.type != 'cpu':
            image_tensor = image_tensor.cpu()
        arr = image_tensor.numpy()

        # Get tensor dimensions (assuming NCHW format)
        if len(arr.shape) == 4:  # Batch processing
            n, c, h, w = arr.shape
        else:  # Single image
            n = 1
            c, h, w = arr.shape if len(arr.shape) == 3 else (1, *arr.shape)

        # Generate random images
        images = []
        for i in range(n):
            if c == 1:  # Grayscale
                random_img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
            else:  # Color
                random_img = np.random.randint(0, 256, (h, w, c), dtype=np.uint8)

            img = Image.fromarray(random_img)
            images.append(img)

            # Save if directory provided
            if save_dir:
                img.save(f"{save_dir}/random_image_{i}.png")
        return images
        # return images[0] if n == 1 else images

    def extract(self):
        """
        extract fingerprint samples.
        :return:
        """
        self.logger.info("-> extract fingerprint...")

        adv = Adv(self.model)
        count = 0
        test_x, test_y = [], []
        min, max = self.get_max_min(self.transform)
        for x, y in self.test_loader:
            since = time.time()
            if self.generated_image == "S":
                x, y = adv.General(x, y, precision=self.precision, targeted=self.targeted, steps=self.steps,
                                   box=(min, max), lr=self.lr, alpha=self.alpha, beta=self.beta)
            else:
                random_images = self.tensor_to_random_image(x)
                batch_images = []
                for random_image in random_images:
                    random_image_tensor = self.transform(random_image)
                    batch_images.append(random_image_tensor)
                random_images_tensor = torch.stack([item for item in batch_images])
                x, y = adv.general_random(x, random_images_tensor, y, precision=self.precision, targeted=self.targeted,
                                          steps=self.steps, box=(min, max), lr=self.lr)
            if x is not None:
                test_x.append(x)
                test_y.append(y)
                count += len(x)
                total_time_elapsed = time.time() - since
                print(f"result_count={count} time_generate={total_time_elapsed}s")
                if count >= self.test_size:
                    break

        fingerprint = {
            "test_x": torch.cat(test_x)[:self.test_size],
            "test_y": torch.cat(test_y)[:self.test_size]
        }
        dest_image_path = osp.join(self.out_root, 'data.pt')
        torch.save(fingerprint, dest_image_path)
        return fingerprint


def create_dir(dir_path):
    if not osp.exists(dir_path):
        print('Path {} does not exist. Creating it...'.format(dir_path))
        os.makedirs(dir_path)


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
    parser.add_argument("--generated_image", default="L", type=str, help="S:source R:random", choices=["S", "R"])
    parser.add_argument('--device_id', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument("--seed", default=1000, type=int, help="Default seed of numpy/pyTorch")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--steps', type=int, default=100, help='iteration steps')
    parser.add_argument('--alpha', type=float, default=1, help='adjustment parameter for value loss ')
    parser.add_argument('--beta', type=float, default=0.00001, help='adjustment parameter for grad loss ')
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
    set_default_seed(args.seed)

    precision = params['precision']
    since = time.time()
    lr = params['lr']
    steps = params['steps']
    alpha = params['alpha']
    beta = params['beta']
    targeted = params['targeted']
    generated_image = params['generated_image']
    general = General(model=model, test_loader=dataloader, transform=transform, device=device, out_root=out_path,
                      max_count=max_count, targeted=targeted, precision=precision, seed=args.seed, lr=lr, steps=steps,
                      alpha=alpha, beta=beta, generated_image=generated_image)
    fingerprint = general.extract()
    total_time_elapsed = time.time() - since
    calc_time = f'{total_time_elapsed // 60:.0f}m {total_time_elapsed % 60:.0f}s'
    params['created_on'] = str(datetime.now())
    params['calc_time'] = calc_time
    params_out_path = osp.join(out_path, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == "__main__":
    main()