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
from utils import revert_tensor_to_image, image_superpixel, save_superpixel, generate_random_mask, \
        masks_to_transformed_tensor, group_attribution_mask, expand_basis_fun

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--datapath', metavar='DIR', nargs='?', default='/mnt/sda1/user/ImageNet/val/',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--split_start', type=int, default=0, help='accelerate', required=False)
parser.add_argument('--split_end', type=int, default=0, help='accelerate', required=False)
parser.add_argument('--subspace_limit', type=int, default=0, help='subspace_limit')
parser.add_argument('--degree', type=int, default=2, help='degree')
parser.add_argument('--samples_min', type=int, default=1000, help='samples_min')
parser.add_argument('--model', type=str, default='resnet18', help='model')

parser.add_argument('--n_superpixels', type=int, default=6, help='n_superpixels')


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
os.makedirs(f"{args.model}_harmonica2degree", exist_ok=True)


for test_index in pbar:
    os.makedirs(f"{args.model}_harmonica2degree/{test_index}", exist_ok=True)

    data, label = test_data.__getitem__(test_index)  # data:tensor, label: int
    data = data.to(device).unsqueeze(0)  # create batch dim
    baseline = torch.zeros_like(data).to(device)

    output = model(data)
    predict = torch.argmax(output, axis=1)

    image = revert_tensor_to_image(data.squeeze(0))
    superpixel = image_superpixel(image, n_superpixels=args.n_superpixels)
    num_of_superpixels = superpixel.max().item()

    n_samples = min(args.samples_min, 2 ** num_of_superpixels)
    basis = generate_random_mask(args,
                                 data,
                                 n_samples=n_samples,
                                 length=num_of_superpixels,
                                 subspace_limit=0)  # 1s and -1s

    masks_tensor = torch.from_numpy((basis + 1) / 2).cuda().bool()

    # process model f output
    masked_samples_tensor = masks_to_transformed_tensor(masks_tensor, data, superpixel)

    masked_samples_dataset = TensorDataset(masked_samples_tensor.to(device))
    masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=64, shuffle=False)

    values = []
    for _data in masked_samples_data_loader:
        values.append(torch.max(model(_data[0]).detach().cpu(), dim=1)[0])
    values = torch.cat(values).squeeze().numpy()

    basis = np.array(basis)

    expanded_basis = expand_basis_fun(basis, args.degree)

    LassoSolver = linear_model.Lasso(fit_intercept=True, alpha=0.01)
    LassoSolver.fit(expanded_basis, values)

    coef = LassoSolver.coef_

    p_bar_info = ""
    for subspace_limit in [0, 1, 2, 4, 8]:

        args.subspace_limit = subspace_limit

        truthful_sample_basis = generate_random_mask(args,
                                                     data,
                                                     n_samples=n_samples,
                                                     length=num_of_superpixels,
                                                     subspace_limit=args.subspace_limit)  # 1s and -1s

        truthful_sample_masks = torch.from_numpy((truthful_sample_basis + 1) / 2).cuda().bool()

        # process model f output
        masked_samples_tensor = masks_to_transformed_tensor(truthful_sample_masks, data, superpixel)

        masked_samples_dataset = TensorDataset(masked_samples_tensor.to(device))
        masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=64, shuffle=False)

        truthful_values = []
        for _data in masked_samples_data_loader:
            truthful_values.append(torch.max(model(_data[0]).detach().cpu(), dim=1)[0])
        model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

        expanded_truthful_sample_basis = expand_basis_fun(truthful_sample_basis, args.degree)

        scikit_lasso_result = (
                np.matmul(expanded_truthful_sample_basis, coef.reshape(-1, 1)) + LassoSolver.intercept_).reshape(-1)
        p_bar_info = p_bar_info + f"{subspace_limit} {np.mean(np.abs(scikit_lasso_result - model_truthful_values))} "

        np.save(
            f"{args.model}_harmonica2degree/{test_index}/testindex{test_index}_final_lasso_output_subspace{subspace_limit}_seed{args.seed}",
            scikit_lasso_result)
        np.save(
            f"{args.model}_harmonica2degree/{test_index}/testindex{test_index}_final_model_output_subspace{subspace_limit}_seed{args.seed}",
            model_truthful_values)

    pbar.set_description("superpixel number: %d" % (num_of_superpixels))

    # loop for C
    for C in C_range:
        x = np.arange(num_of_superpixels)
        combination_mask_index_under_degree = []
        truthful_degree = np.arange(0, C + 1).astype(int)
        for each_limit in truthful_degree:
            combination_mask_index_under_degree += list(itertools.combinations(x, each_limit))
        combination_mask = np.ones((len(combination_mask_index_under_degree), num_of_superpixels))
        for i, each_mask in enumerate(combination_mask_index_under_degree):
            for index in each_mask:
                combination_mask[i][index] = 0

        combination_mask = (1 - combination_mask)  # number of 1s <= 3, this is kai_s

        # loop for subspace
        for subspace_limit in [0, 1, 2, 4, 8]:
            args.subspace_limit = subspace_limit
            truthful_sample_basis = generate_random_mask(args,
                                                         data,
                                                         n_samples=n_samples,
                                                         length=num_of_superpixels,
                                                         subspace_limit=args.subspace_limit)  # 1s and -1s
            # calculate sigma
            sigmas = []
            for each_x in truthful_sample_basis:
                for each_combination_mask in combination_mask:
                    if len(np.where(each_combination_mask * each_x == -1)[0]) % 2 == 0:
                        _sum = 1
                    else:
                        _sum = -1
                sigmas.append(_sum)
            sigmas = np.array(sigmas)

            # process model f output
            truthful_sample_masks = torch.from_numpy((truthful_sample_basis + 1) / 2).cuda().bool()
            masked_samples_tensor = masks_to_transformed_tensor(truthful_sample_masks, data, superpixel)

            masked_samples_dataset = TensorDataset(masked_samples_tensor.to(device))
            masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=64, shuffle=False)

            truthful_values = []
            for _data in masked_samples_data_loader:
                truthful_values.append(torch.max(model(_data[0]).detach().cpu(), dim=1)[0])
            model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

            expanded_truthful_sample_basis = expand_basis_fun(truthful_sample_basis, args.degree)

            scikit_lasso_result = (
                    np.matmul(expanded_truthful_sample_basis,
                              coef.reshape(-1, 1)) + LassoSolver.intercept_).reshape(-1)

            answer = np.mean((model_truthful_values - scikit_lasso_result) * sigmas)

            np.save(
                f"{args.model}_harmonica2degree/{test_index}/testindex{test_index}_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                scikit_lasso_result)
            np.save(
                f"{args.model}_harmonica2degree/{test_index}/testindex{test_index}_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                model_truthful_values)
            np.save(
                f"{args.model}_harmonica2degree/{test_index}/testindex{test_index}_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                sigmas)
            np.save(
                f"{args.model}_harmonica2degree/{test_index}/testindex{test_index}_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                answer)

    torch.cuda.empty_cache()

