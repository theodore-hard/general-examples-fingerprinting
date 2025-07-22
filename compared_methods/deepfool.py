import copy
# from torch.autograd.gradcheck import zero_gradients
import argparse
import collections
import copy
import json
import os
import os.path as osp
import time
from datetime import datetime
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from fingerprint import datasets
from fingerprint.victim.blackbox import Blackbox

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):
    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0, I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_image


def main():
    parser = argparse.ArgumentParser(description="Build micro benchmark.")
    parser.add_argument("dataset_name", type=str, help="extract dataset name")
    parser.add_argument("dataset_type", type=str, help="extract dataset type")
    parser.add_argument('victim_model_dir', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('out_path', type=str, help='Output path for fingerprints')
    parser.add_argument('max_count', type=int, help='Output image max count')
    parser.add_argument('--device_id', type=int, help='Device id. -1 for CPU.', default=0)
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
    since = time.time()
    count = 0
    test_x, test_y = [], []
    for x, y in dataloader:
        r, loop_i, label_orig, label_pert, pert_image = deepfool(x[0], model)
        test_x.append(pert_image.cpu().detach())
        test_y.append(torch.tensor([label_pert]))
        count += len(x)
        if count >= max_count:
            break
    fingerprint = {
        "test_x": torch.cat(test_x)[:max_count],
        "test_y": torch.cat(test_y)[:max_count]
    }
    dest_image_path = osp.join(out_path, 'data.pt')
    torch.save(fingerprint, dest_image_path)

    total_time_elapsed = time.time() - since
    calc_time = f'{total_time_elapsed // 60:.0f}m {total_time_elapsed % 60:.0f}s'
    params['created_on'] = str(datetime.now())
    params['calc_time'] = calc_time
    params_out_path = osp.join(out_path, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == "__main__":
    main()
