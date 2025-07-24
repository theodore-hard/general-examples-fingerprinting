#!/usr/bin/python
import os.path as osp
from torchvision.datasets import ImageFolder
import config as cfg


class Cifar10(ImageFolder):
    def __init__(self, period, transform=None, target_transform=None):
        root = osp.join(cfg.DATASET_ROOT, 'cifar10')
        if not osp.exists(root):
            raise ValueError('Dataset not found:' + root)
        _root = osp.join(root, period)
        super().__init__(root=_root, transform=transform,
                         target_transform=target_transform)
        self.root = root
        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, period, len(self.samples)))


class Cifar100(ImageFolder):
    def __init__(self, period, transform=None, target_transform=None):
        root = osp.join(cfg.DATASET_ROOT, 'cifar100')
        if not osp.exists(root):
            raise ValueError('Dataset not found:' + root)
        _root = osp.join(root, period)
        super().__init__(root=_root, transform=transform,
                         target_transform=target_transform)
        self.root = root
        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, period, len(self.samples)))


class ImageNet(ImageFolder):
    def __init__(self, period, transform=None, target_transform=None):
        root = osp.join(cfg.DATASET_ROOT, 'imagenet')
        if not osp.exists(root):
            raise ValueError('Dataset not found:' + root)
        _root = osp.join(root, period)
        super().__init__(root=_root, transform=transform,
                         target_transform=target_transform)
        self.root = root
        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, period, len(self.samples)))
