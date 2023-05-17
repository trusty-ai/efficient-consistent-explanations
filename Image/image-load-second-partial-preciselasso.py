import os
import torch
import random
import argparse
import itertools
import torchvision

import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models
from sklearn import linear_model
from utils import transform
from utils import revert_tensor_to_image, image_superpixel, save_superpixel, \
        masks_to_transformed_tensor, group_attribution_mask, expand_basis_fun

from scipy.special import comb
from itertools import product
from collections import Counter

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--datapath', metavar='DIR', nargs='?', default='/mnt/sda1/user/ImageNet/val/',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--split_start', type=int, default=0, help='accelerate', required=False)
parser.add_argument('--split_end', type=int, default=0, help='accelerate', required=False)
parser.add_argument('--subspace_limit', type=int, default=0, help='subspace_limit')
parser.add_argument('--degree', type=int, default=2, help='degree')
parser.add_argument('--samples_min', type=int, default=1000, help='samples_min')
parser.add_argument('--n', type=int, default=3, help='n anchors')
parser.add_argument('--ep_consistent_loss', type=float, default=0, help='ep_consistent_loss')
parser.add_argument('--n_superpixels', type=int, default=6, help='n_superpixels')
parser.add_argument('--model', type=str, default='resnet18', help='model')



args = parser.parse_args()

SEED = args.seed

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_data = torchvision.datasets.ImageFolder(args.datapath, transform=transform)

random_index = np.arange(len(test_data))
np.random.shuffle(random_index)

# imagenet_validset_subset = torch.utils.data.Subset(test_data, random_index)
# subset_dataloader = DataLoader(imagenet_validset_subset, batch_size=1)
# print(len(imagenet_validset_subset))

if args.model == "resnet18":
    model = models.resnet18(pretrained=True)
elif args.model == "resnet101":
    model = models.resnet101(pretrained=True)
else:
    raise NotImplementedError

model = nn.Sequential(
    model,
    nn.Softmax(1)
)

model = model.to(device)
model = model.eval()

if args.split_end - args.split_start == 0 and args.split_start == 0:
    pbar = tqdm(random_index)
else:
    pbar = tqdm(random_index[args.split_start: args.split_end])

C_range = [1, 2, 3]
os.makedirs(f"{args.model}_harmonica2degreepartial_preciselasso_sample_sample{args.samples_min}_anchor{args.n}_consisloss{args.ep_consistent_loss}", exist_ok=True)

def generate_random_mask(args, x, n_samples=1000, length=0, subspace_limit=0, anchor=None):
    # should return 1 and -1
    assert x.shape[0] == 1  # default: batch size
    if subspace_limit > length:
        subspace_limit = length
    assert subspace_limit <= length, f"{subspace_limit, length}"  # maximum number of indexes of 0s

    if subspace_limit == 0:
        if n_samples == args.samples_min:
            mask_matrix = ((np.random.rand(n_samples, length) > .5) * 2 - 1).astype(int)
        else:
            mask_matrix = np.array(list(product([-1, 1], repeat=length)))
    else:  # subspace_limit is not 0
        if n_samples == args.samples_min:
            combnition_number_list = []
            for i in range(subspace_limit, 0, -1):
                comb_num = comb(length, i)
                if len(combnition_number_list) == 0 or comb_num / combnition_number_list[0] > 1 / n_samples:
                    combnition_number_list.append(comb_num)
            combnition_number_prob = combnition_number_list / sum(combnition_number_list)
            num_of_zeros = np.random.choice(
                np.arange(subspace_limit, subspace_limit - len(combnition_number_list), -1), n_samples,
                p=combnition_number_prob)
            column_index_every_row = [np.random.choice(length, num_of_zero, replace=False) for num_of_zero in
                                      num_of_zeros]

            mask_matrix = np.ones((n_samples, length))
            for _i in range(n_samples):
                mask_matrix[_i, column_index_every_row[_i]] = 0
            mask_matrix = mask_matrix * 2 - 1
        else:
            mask_matrix = np.array(list(product([0, 1], repeat=length)))
            mask_matrix = mask_matrix[np.where(mask_matrix.sum(axis=1) >= length - subspace_limit)[0], :].squeeze()
            mask_matrix = mask_matrix * 2 - 1

    if anchor is not None:  # ensure ther are at least 1 basis assigned to each anchor
        mask_matrix = np.vstack([anchor, mask_matrix])

    return mask_matrix

def generate_random_anchor(n_minus_1, num_of_superpixels):
    length = num_of_superpixels
    anchor_matrix = ((np.random.rand(n_minus_1, length) > .5) * 2 - 1).astype(int)
    anchor_matrix = np.vstack([np.ones([1, length]), anchor_matrix])

    return anchor_matrix

def generate_local_mask(args, num_of_superpixels, n_samples=1000, subspace_limit=0):
    # should return 1 and -1
    length = num_of_superpixels
    if subspace_limit > length:
        subspace_limit = length
    assert subspace_limit <= length, f"{subspace_limit, length}"  # maximum number of indexes of 0s

    if n_samples == int(args.samples_min / args.n):
        combnition_number_list = []
        for i in range(subspace_limit, 0, -1):
            comb_num = comb(length, i)
            if len(combnition_number_list)==0 or comb_num / combnition_number_list[0] > 1 / n_samples:
                combnition_number_list.append(comb_num)
        combnition_number_prob = combnition_number_list / sum(combnition_number_list)
        num_of_zeros = np.random.choice(np.arange(subspace_limit, subspace_limit - len(combnition_number_list), -1), n_samples, p=combnition_number_prob)
        column_index_every_row = [np.random.choice(length, num_of_zero, replace=False) for num_of_zero in num_of_zeros]

        mask_matrix = np.ones((n_samples, length))
        for _i in range(n_samples):
            mask_matrix[_i, column_index_every_row[_i]] = 0
        mask_matrix = mask_matrix * 2 - 1
    else:
        mask_matrix = np.array(list(product([0, 1], repeat=length)))
        mask_matrix = mask_matrix[np.where(mask_matrix.sum(axis=1) >= length-subspace_limit)[0], :].squeeze()
        mask_matrix = mask_matrix * 2 - 1

    return np.float32(mask_matrix)

def assign_basis_to_anchor(basis, anchor_matrix, limit_set=None):
    anchor_index_list = []
    # compute distance
    for each_basis in basis:
        each_distance = np.sum(np.abs(anchor_matrix - each_basis), axis=1)
        if limit_set is not None:
            for _i in np.argsort(each_distance):
                each_index = np.min(np.where(each_distance == each_distance[_i])[0])
                if each_index in limit_set:
                    anchor_index_list.append(each_index)
                    break
        else:
            each_index = np.min(np.where(each_distance == each_distance.min())[0])
            anchor_index_list.append(each_index)
    return np.array(anchor_index_list)


for test_index in pbar:
    os.makedirs(f"{args.model}_harmonica2degreepartial_preciselasso_sample_sample{args.samples_min}_anchor{args.n}_consisloss{args.ep_consistent_loss}/{test_index}", exist_ok=True)

    data, label = test_data.__getitem__(test_index)  # data:tensor, label: int
    data = data.to(device).unsqueeze(0)  # create batch dim
    baseline = torch.zeros_like(data).to(device)

    output = model(data)
    predict = torch.argmax(output, axis=1)

    image = revert_tensor_to_image(data.squeeze(0))
    superpixel = image_superpixel(image, n_superpixels=args.n_superpixels)
    num_of_superpixels = superpixel.max().item()

    n_samples = min(args.samples_min, 2 ** num_of_superpixels)

    anchor = generate_random_anchor(args.n-1, num_of_superpixels)

    local_basis_sample_number = min(comb(num_of_superpixels, 4), n_samples / args.n)
    local_basis = generate_local_mask(args,
                                      num_of_superpixels,
                                      n_samples=int(local_basis_sample_number),
                                      subspace_limit=2
                                      )

    basis = generate_random_mask(args,
                                 data,
                                 n_samples=n_samples,
                                 length=num_of_superpixels,
                                 subspace_limit=0)  # 1s and -1s
    basis = np.vstack([local_basis, basis])

    anchor_index_list = assign_basis_to_anchor(basis, anchor)  # np.array
    used_n = list(Counter(anchor_index_list).keys())

    masks_tensor = torch.from_numpy((basis + 1) / 2).cuda().bool()

    # process model f output
    masked_samples_tensor = masks_to_transformed_tensor(masks_tensor, data, superpixel)

    masked_samples_dataset = TensorDataset(masked_samples_tensor.to(device))
    masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=32, shuffle=False)

    values = []
    for _data in masked_samples_data_loader:
        values.append(torch.max(model(_data[0]).detach().cpu(), dim=1)[0])
    values = torch.cat(values).squeeze().numpy()

    basis = np.array(basis)

    expanded_basis = expand_basis_fun(basis, args.degree)
    expanded_basis = np.hstack([np.ones((expanded_basis.shape[0], 1)), expanded_basis])

    anchor_params = np.zeros((args.n, expanded_basis.shape[1]))
    anchor_intercepts = np.zeros((args.n, 1))

    for _anchor_i in used_n:
        _anchor_basis_index = np.where(np.array(anchor_index_list)==_anchor_i)
        _anchor_expanded_basis = expanded_basis[_anchor_basis_index]
        _anchor_values = np.array(values)[_anchor_basis_index]
        LassoSolver = linear_model.Lasso(fit_intercept=True, alpha=0.001)
        LassoSolver.fit(_anchor_expanded_basis, _anchor_values)
        _anchor_coef = LassoSolver.coef_
        anchor_params[_anchor_i] = _anchor_coef
        anchor_intercepts[_anchor_i] = LassoSolver.intercept_

    p_bar_info = ""
    for subspace_limit in [0, 1, 2, 4, 8]:

        args.subspace_limit = subspace_limit

        truthful_sample_basis = generate_random_mask(args,
                                                     data,
                                                     n_samples=n_samples,
                                                     length=num_of_superpixels,
                                                     subspace_limit=args.subspace_limit)  # 1s and -1s
        truthful_sample_anchor_index_list = assign_basis_to_anchor(truthful_sample_basis, anchor, limit_set=used_n)  # np.array

        truthful_sample_masks = torch.from_numpy((truthful_sample_basis + 1) / 2).cuda().bool()

        # process model f output
        masked_samples_tensor = masks_to_transformed_tensor(truthful_sample_masks, data, superpixel)

        masked_samples_dataset = TensorDataset(masked_samples_tensor.to(device))
        masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=32, shuffle=False)

        truthful_values = []
        for _data in masked_samples_data_loader:
            truthful_values.append(torch.max(model(_data[0]).detach().cpu(), dim=1)[0])
        model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

        expanded_truthful_sample_basis = expand_basis_fun(truthful_sample_basis, args.degree)
        expanded_truthful_sample_basis = np.hstack([np.ones((expanded_truthful_sample_basis.shape[0], 1)), expanded_truthful_sample_basis])

        scikit_lasso_result = []
        for _i, each_index in enumerate(truthful_sample_anchor_index_list):
            assert each_index in used_n
            # assert np.sum(anchor_params[each_index]) != 0
            scikit_lasso_result.append(
                np.sum(expanded_truthful_sample_basis[_i] * anchor_params[each_index]) + anchor_intercepts[each_index])
        scikit_lasso_result = np.array(scikit_lasso_result).reshape(-1)

        p_bar_info = p_bar_info + f"{subspace_limit} {np.mean(np.abs(scikit_lasso_result - model_truthful_values))} "

        np.save(
            f"{args.model}_harmonica2degreepartial_preciselasso_sample_sample{args.samples_min}_anchor{args.n}_consisloss{args.ep_consistent_loss}/{test_index}/testindex{test_index}_final_lasso_output_subspace{subspace_limit}_seed{args.seed}",
            scikit_lasso_result)
        np.save(
            f"{args.model}_harmonica2degreepartial_preciselasso_sample_sample{args.samples_min}_anchor{args.n}_consisloss{args.ep_consistent_loss}/{test_index}/testindex{test_index}_final_model_output_subspace{subspace_limit}_seed{args.seed}",
            model_truthful_values)

    pbar.set_description("superpixel number: %d" % (num_of_superpixels))


"""
CUDA_VISIBLE_DEVICES=0 python image-load-second-partial-preciselasso.py --model resnet101 --n_superpixels 16 --samples_min 1000 --n 9 --ep_consistent_loss 0 --split_start 0 --split_end 10

"""
