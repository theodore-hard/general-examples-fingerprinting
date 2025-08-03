# general-examples-fingerprinting


## Introduction

This is a [PyTorch](https://pytorch.org/) implementation of [Deep Neural Network Fingerprinting By General Examples]().  
The main reason to develop this respository is to make it easier to do research using the attach technique.  
The project has been tested under `python 3.9.10` and `torch-1.10.2`.

## References:

- [tribhuvanesh/knockoffnets](https://github.com/tribhuvanesh/knockoffnets)
- [grasses/RemovalNet](https://github.com/grasses/RemovalNet)
## Usage of this library module

### trained model
Training from scratch: You can train the model using the provided train.py script. The usage method is:
```bash
train.py Cifar10 googlenet -o /project/general-examples-fingerprinting-main/outputs/victim/cifar10-googlenet -e 256 --lr 0.1 --lr-step 100 --lr-gamma 0.1
```
Using a pre-trained model: We provide a download.py script to download and encapsulate PyTorch pre-trained models. The usage method is:
```bash
download.py ImageNet vgg11 -o /project/general-examples-fingerprinting-main/outputs/victim/imagenet-vgg11 --pretrained true
```
### extract general examples fingerprints
Using extract.py, input necessary parameters such as the model and sample data. The usage method is:
```bash
extract.py Cifar10 train  /project/general-examples-fingerprinting-main/outputs/victim/cifar10-googlenet /project/general-examples-fingerprinting-main/outputs/fingerprint/cifar10-googlenet/general/0.999  100  --precision 0.999 --lr 0.01  --steps 100
```
### verify fingerprints
Using verify.py, input the fingerprint data and the model to be verified. The usage method is:
```bash
verify.py /project/general-examples-fingerprinting-main/outputs/victim/cifar10-googlenet /project/general-examples-fingerprinting-main/outputs/fingerprint/cifar10-googlenet/general/0.999
```


