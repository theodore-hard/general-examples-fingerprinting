#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import json
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

import models.zoo as zoo


class TypeCheck(object):
    @staticmethod
    def single_image_blackbox_input(bb_input):
        if not isinstance(bb_input, np.ndarray):
            raise TypeError("Input must be an np.ndarray.")
        if bb_input.dtype != np.uint8:
            raise TypeError("Input must be an unit8 array.")
        if len(bb_input.shape) != 3:
            raise TypeError("Input must have three dims with (channel, height, width) elements.")

    @staticmethod
    def multiple_image_blackbox_input(bb_input):
        if not isinstance(bb_input, np.ndarray):
            raise TypeError("Input must be an np.ndarray.")
        if bb_input.dtype != np.uint8:
            raise TypeError("Input must be an unit8 array.")
        if len(bb_input.shape) != 4:
            raise TypeError("Input must have three dims with (num_samples, channel, height, width) elements.")

    @staticmethod
    def multiple_image_blackbox_input_tensor(bb_input):
        if not isinstance(bb_input, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if bb_input.dtype != torch.float32:
            raise TypeError("Input must be a torch.float32 tensor.")
        if len(bb_input.shape) != 4:
            raise TypeError("Input must have three dims with (num_samples, channel, height, width) elements.")

    @staticmethod
    def single_label_int(label):
        if not isinstance(label, int):
            raise TypeError("Label must be an int.")

    @staticmethod
    def multiple_label_list_int(labels):
        if not isinstance(labels, list):
            raise TypeError("Labels must be a list.")
        for l in labels:
            if not isinstance(l, int):
                raise TypeError("Each label must be an int.")

class Blackbox(object):
    def __init__(self, model, device=None, output_type='probs', topk=None, rounding=None, num_classes=None,
                 model_arch=None):
        self.device = torch.device('cuda') if device is None else device
        self.output_type = output_type
        self.topk = topk
        self.rounding = rounding
        self.num_classes = num_classes
        self.model_arch = model_arch

        self.__model = model.to(device)
        self.__model.eval()

        self.__call_count = 0

    @classmethod
    def from_modeldir(cls, model_dir, device=None, output_type='probs'):
        device = torch.device('cuda') if device is None else device

        # What was the model architecture used by this model?
        params_path = osp.join(model_dir, 'params.json')
        if not os.path.exists(params_path):
            params_dir = os.path.dirname(os.path.dirname(params_path))
            params_path = osp.join(params_dir, 'params.json')
        with open(params_path) as jf:
            params = json.load(jf)
        model_arch = params['model_arch']
        num_classes = params['num_classes']
        model = zoo.get_net(model_arch, num_classes=num_classes)
        model = model.to(device)

        # Load weights
        checkpoint_path = osp.join(model_dir, 'model_best.pth.tar')
        if not osp.exists(checkpoint_path):
            checkpoint_path = osp.join(model_dir, 'checkpoint.pth.tar')
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        blackbox = cls(model, device=device, output_type=output_type, num_classes=num_classes, model_arch=model_arch)
        return blackbox

    @classmethod
    def from_modeldir_with_lastlayer_changed(cls, model_dir, device=None, output_type='probs'):
        device = torch.device('cuda') if device is None else device

        # What was the model architecture used by this model?
        params_path = osp.join(model_dir, 'params.json')
        if not os.path.exists(params_path):
            params_dir = os.path.dirname(os.path.dirname(params_path))
            params_path = osp.join(params_dir, 'params.json')
        with open(params_path) as jf:
            params = json.load(jf)
        model_arch = params['model_arch']
        num_classes = params['num_classes']
        source_model_num_classes = params['source_model_num_classes']
        model = zoo.get_net(model_arch, num_classes=source_model_num_classes)
        if isinstance(model, torchvision.models.ResNet) or isinstance(model, torchvision.models.GoogLeNet):
            num_ftrs = model.fc.in_features  # 获取原网络中的特征输入
            model.fc = torch.nn.Linear(num_ftrs, num_classes)
        elif isinstance(model, torchvision.models.DenseNet):
            num_ftrs = model.classifier.in_features  # 获取原网络中的特征输入
            model.classifier = torch.nn.Linear(num_ftrs, num_classes)
        elif isinstance(model, torchvision.models.AlexNet) or isinstance(model, torchvision.models.VGG):
            num_features = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(num_features, num_classes)
        else:
            num_ftrs = model.classifier.in_features  # 获取原网络中的特征输入
            model.classifier = torch.nn.Linear(num_ftrs, num_classes)
        model = model.to(device)

        # Load weights
        checkpoint_path = osp.join(model_dir, 'model_best.pth.tar')
        if not osp.exists(checkpoint_path):
            checkpoint_path = osp.join(model_dir, 'checkpoint.pth.tar')
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {}, acc={:.2f})".format(epoch, best_test_acc))

        blackbox = cls(model, device=device, output_type=output_type, num_classes=num_classes, model_arch=model_arch)
        return blackbox

    def get_model(self):
        return self.__model

    def truncate_output(self, y_t_probs):
        if self.topk is not None:
            # Zero-out everything except the top-k predictions
            topk_vals, indices = torch.topk(y_t_probs, self.topk)
            newy = torch.zeros_like(y_t_probs)
            if self.rounding == 0:
                # argmax prediction
                newy = newy.scatter(1, indices, torch.ones_like(topk_vals))
            else:
                newy = newy.scatter(1, indices, topk_vals)
            y_t_probs = newy

        # Rounding of decimals
        if self.rounding is not None:
            y_t_probs = torch.Tensor(np.round(y_t_probs.numpy(), decimals=self.rounding))

        return y_t_probs

    def train(self):
        raise ValueError('Cannot run blackbox model in train mode')

    def eval(self):
        # Always in eval mode
        pass

    def get_call_count(self):
        return self.__call_count

    def __call__(self, query_input):
        TypeCheck.multiple_image_blackbox_input_tensor(query_input)

        with torch.no_grad():
            query_input = query_input.to(self.device)
            query_output = self.__model(query_input)

            if isinstance(query_output, tuple):
                query_output = query_output[0]

            self.__call_count += query_input.shape[0]

            query_output_probs = F.softmax(query_output, dim=1)

        query_output_probs = self.truncate_output(query_output_probs)
        return query_output_probs
