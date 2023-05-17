import os
import torch
import random
import argparse
import itertools
import torchvision

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from scipy.special import comb
from torchvision import models
from torchtext.legacy import data
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from utils import transform

from torch.autograd.functional import hessian
from torch.autograd.functional import jacobian

parser = argparse.ArgumentParser(description='consistent args')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--datapath', metavar='DIR', nargs='?', default='/mnt/sda1/user/ImageNet/val/',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--subspace_limit', type=int, default=0, help='subspace_limit')
parser.add_argument('--samples_min', type=int, default=2000, help='samples_min')

parser.add_argument('--algorithm', type=str, default="ih", help='algorithm name')

parser.add_argument('--split_start', type=int, default=0, help='accelerate', required=False)
parser.add_argument('--split_end', type=int, default=0, help='accelerate', required=False)
parser.add_argument('--model', type=str, default='resnet18', help='model')

parser.add_argument('--n_superpixels', type=int, default=6, help='n_superpixels')

args = parser.parse_args()

assert args.algorithm in ["ih", "shaptaylor", "faithshap"]

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


pbar = tqdm(range((len(test_data))))

if args.split_end - args.split_start == 0 and args.split_start == 0:
    pbar = tqdm(random_index)
else:
    assert args.algorithm in ["ih", "shaptaylor", "faithshap"], "Other algorithms do not support acceleration"
    pbar = tqdm(random_index[args.split_start: args.split_end])


if args.algorithm == "ih":
    print("start ih")

    os.makedirs(f"{args.model}_ih", exist_ok=True)
    from utils import revert_tensor_to_image, image_superpixel, save_superpixel, generate_random_mask, \
        masks_to_transformed_tensor, group_attribution_mask

    def integrated_hessians(func, dim):
        samples = 50
        alphas = torch.rand(samples)
        betas = torch.rand(samples)
        alpha_betas = alphas * betas
        acc_hessian = torch.zeros(dim, dim, device="cuda")
        acc_grad = torch.zeros(dim, device="cuda")
        weight = 1 / (samples)
        for alpha_beta in alpha_betas:
            blended_x = Variable(torch.ones(dim, device="cuda") * alpha_beta, requires_grad=True)
            acc_hessian += hessian(func, blended_x).squeeze() * (alpha_beta * weight)
            acc_grad += jacobian(func, blended_x).squeeze() * weight
        return acc_hessian, acc_grad

    # init variables
    C_range = [1, 2, 3]
    for test_index in pbar:
        os.makedirs(f"{args.model}_ih/{test_index}", exist_ok=True)

        data, label = test_data.__getitem__(test_index)  # data:tensor, label: int
        data = data.to(device).unsqueeze(0)  # create batch dim
        baseline = torch.zeros_like(data).to(device)

        output = model(data)
        predict = torch.argmax(output, axis=1)

        image = revert_tensor_to_image(data.squeeze(0))
        superpixel = image_superpixel(image, n_superpixels=args.n_superpixels)
        superpixel_one_hot = torch.nn.functional.one_hot(superpixel - torch.min(superpixel)).to(device)
        num_of_superpixels = superpixel.max().item()

        def func(blended_mask):
            mask_q = torch.sum(superpixel_one_hot * blended_mask.reshape(1, 1, -1), dim=2)
            mask_q = mask_q.reshape(1, 1, 224, 224)
            result = data * mask_q + baseline * (1 - mask_q)
            result = model(result)[0, predict]

            return result

        acc_hessian, acc_grad = integrated_hessians(func, dim=num_of_superpixels)

        acc_hessian = acc_hessian.reshape(-1, 1).cpu().numpy()
        acc_grad = acc_grad.reshape(-1, 1).cpu().numpy()
        incompleteness = model(data)[0, predict] - model(baseline)[0, predict] - acc_hessian.sum() - acc_grad.sum()
        incompleteness = incompleteness.detach().cpu().numpy()

        p_bar_info = ""
        for subspace_limit in [0, 1, 2, 4, 8]:

            args.subspace_limit = subspace_limit

            n_samples = min(args.samples_min, 2 ** num_of_superpixels)
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

            truthful_sample_masks = truthful_sample_masks.int()
            second_order_masks = []
            for each_mask in truthful_sample_masks:
                tmp_mask = np.ones((each_mask.shape[0], each_mask.shape[0]))
                each_mask_zero_index = torch.where(each_mask==0)
                for index in each_mask_zero_index[0]:
                    tmp_mask[index].fill(0)
                    tmp_mask[:,index].fill(0)
                second_order_masks.append(tmp_mask)
            second_order_masks = np.stack(second_order_masks, axis=0)

            # flatten
            second_order_masks = second_order_masks.reshape(second_order_masks.shape[0], -1)
            first_order_masks = truthful_sample_masks.reshape(truthful_sample_masks.shape[0], -1)

            ih_result = (
                    np.matmul(second_order_masks, acc_hessian)
                    + np.matmul(first_order_masks.cpu().numpy(), acc_grad)
                    + model(baseline)[0, predict].detach().cpu().numpy()
                    + incompleteness
            ).reshape(-1)

            p_bar_info = p_bar_info + f"{subspace_limit} {np.mean(np.abs(ih_result - model_truthful_values))} "

            np.save(
                f"{args.model}_ih/{test_index}/testindex{test_index}_final_lasso_output_subspace{subspace_limit}_seed{args.seed}",
                ih_result)
            np.save(
                f"{args.model}_ih/{test_index}/testindex{test_index}_final_model_output_subspace{subspace_limit}_seed{args.seed}",
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

                masked_samples_dataset = TensorDataset(masked_samples_tensor)
                masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=64, shuffle=False)

                truthful_values = []
                for _data in masked_samples_data_loader:
                    truthful_values.append(torch.max(model(_data[0]).detach().cpu(), dim=1)[0])
                model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

                truthful_sample_masks = truthful_sample_masks.int()
                second_order_masks = []
                for each_mask in truthful_sample_masks:
                    tmp_mask = np.ones((each_mask.shape[0], each_mask.shape[0]))
                    each_mask_zero_index = torch.where(each_mask == 0)
                    for index in each_mask_zero_index[0]:
                        tmp_mask[index].fill(0)
                        tmp_mask[:, index].fill(0)
                    second_order_masks.append(tmp_mask)
                second_order_masks = np.stack(second_order_masks, axis=0)

                # flatten
                second_order_masks = second_order_masks.reshape(second_order_masks.shape[0], -1)
                first_order_masks = truthful_sample_masks.reshape(truthful_sample_masks.shape[0], -1)

                ih_result = (
                        np.matmul(second_order_masks, acc_hessian)
                        + np.matmul(first_order_masks.cpu().numpy(), acc_grad)
                        + model(baseline)[0, predict].detach().cpu().numpy()
                        + incompleteness
                ).reshape(-1)

                answer = np.mean((model_truthful_values - ih_result) * sigmas)

                np.save(
                    f"{args.model}_ih/{test_index}/testindex{test_index}_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    ih_result)
                np.save(
                    f"{args.model}_ih/{test_index}/testindex{test_index}_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    model_truthful_values)
                np.save(
                    f"{args.model}_ih/{test_index}/testindex{test_index}_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    sigmas)
                np.save(
                    f"{args.model}_ih/{test_index}/testindex{test_index}_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    answer)

        torch.cuda.empty_cache()


elif args.algorithm == "shaptaylor":

    os.makedirs(f"{args.model}_shaptaylor", exist_ok=True)
    from torch_lr import TorchRidge
    from utils import revert_tensor_to_image, image_superpixel, generate_random_mask, \
        masks_to_transformed_tensor

    def safe_st_kernel(samples):
        samples = samples[np.where(samples.sum(axis=1) != samples.shape[1])]
        samples = samples[np.where(samples.sum(axis=1) != 0)]
        samples = samples[np.where(samples.sum(axis=1) != 1)]
        s = samples.sum(axis=1)
        M = np.ones(samples.shape[0]) * samples.shape[1]
        weights = (M - 1) / (comb(M, s) * comb(s, 2) * (M - s))
        return weights, samples

    def shap_sampler(dim, shap_n_samples):
        inf_samples = np.concatenate([
            np.ones((1, dim)),
            np.zeros((1, dim)),
        ], axis=0)
        reg_samples = (np.random.rand(shap_n_samples - inf_samples.shape[0], dim) > .5)

        reg_weights, reg_samples = safe_st_kernel(reg_samples)
        reg_weights /= reg_weights.sum()
        reg_weights *= reg_weights.shape[0]
        inf_weights = np.ones(inf_samples.shape[0])
        inf_weights /= inf_weights.shape[0]
        inf_weights *= reg_weights.shape[0] * 3

        weights = np.concatenate([reg_weights, inf_weights], axis=0)
        samples = np.concatenate([reg_samples, inf_samples], axis=0).astype(np.bool)

        return torch.from_numpy(weights).to(device), \
               torch.from_numpy(samples).to(device).int()

    def calculate_second_order_shap_taylor_interaction(model, sample_tensor, baseline_tensor, dim, shap_n_samples, superpixel):

        weights, masks = shap_sampler(dim, shap_n_samples)

        samples_tensor = masks_to_transformed_tensor(masks, sample_tensor, superpixel).cpu()

        dataset = TensorDataset(samples_tensor)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        values = []
        for data in data_loader:
            values.append(torch.max(model(data[0].to(device)).detach().cpu(), dim=1)[0])
        values = torch.cat(values).squeeze().numpy()

        cross_terms = np.apply_along_axis(lambda s: (np.outer(s, s)).reshape(-1), 1, masks.cpu())
        term_mask = np.triu(np.ones((dim, dim), dtype=bool), k=0).reshape(-1)
        terms_to_keep = np.where(term_mask)
        cross_terms = cross_terms[:, terms_to_keep].squeeze()
        features = cross_terms.astype(bool)

        # second order shap taylor
        model = TorchRidge(alpha=.01, fit_intercept=True)
        weights = weights.cpu().numpy()
        model.fit(features, values, weights)

        full_coeff = np.zeros((dim * dim))
        full_coeff[terms_to_keep] = model.coef_[:cross_terms.shape[1]]
        full_coeff = full_coeff.reshape(dim, dim)

        return full_coeff

    # init variables
    C_range = [1, 2, 3]

    for test_index in pbar:
        os.makedirs(f"{args.model}_shaptaylor/{test_index}", exist_ok=True)

        data, label = test_data.__getitem__(test_index)  # data:tensor, label: int
        data = data.to(device).unsqueeze(0)  # create batch dim
        baseline = torch.zeros_like(data).to(device)

        output = model(data)
        predict = torch.argmax(output, axis=1)

        image = revert_tensor_to_image(data.squeeze(0))
        superpixel = image_superpixel(image, n_superpixels=args.n_superpixels)
        superpixel_one_hot = torch.nn.functional.one_hot(superpixel - torch.min(superpixel)).to(device)
        num_of_superpixels = superpixel.max().item()

        shap_n_samples = min(1000, 2**num_of_superpixels)

        full_coeff = calculate_second_order_shap_taylor_interaction(model,
                                                                    data,
                                                                    baseline,
                                                                    num_of_superpixels,
                                                                    shap_n_samples,
                                                                    superpixel,
                                                                    )
        p_bar_info = ""
        for subspace_limit in [0, 1, 2, 4, 8]:
            args.subspace_limit = subspace_limit

            # calculate consistency
            n_samples = min(args.samples_min, 2**num_of_superpixels)
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

            truthful_sample_masks = truthful_sample_masks.int()
            second_order_masks = []
            for each_mask in truthful_sample_masks:
                tmp_mask = np.ones((each_mask.shape[0], each_mask.shape[0]))
                each_mask_zero_index = torch.where(each_mask == 0)
                for index in each_mask_zero_index[0]:
                    tmp_mask[index].fill(0)
                    tmp_mask[:, index].fill(0)
                second_order_masks.append(tmp_mask)
            second_order_masks = np.stack(second_order_masks, axis=0)

            # flatten
            full_coeff = full_coeff.reshape(-1, 1)
            second_order_masks = second_order_masks.reshape(second_order_masks.shape[0], -1)

            shaptaylor_result = (
                np.matmul(second_order_masks, full_coeff) + model(baseline)[0, predict].detach().cpu().numpy() ).reshape(-1)

            p_bar_info = p_bar_info + f"{subspace_limit} {np.mean(np.abs(shaptaylor_result - model_truthful_values))} "

            np.save(
                f"{args.model}_shaptaylor/{test_index}/testindex{test_index}_final_lasso_output_subspace{subspace_limit}_seed{args.seed}",
                shaptaylor_result)
            np.save(
                f"{args.model}_shaptaylor/{test_index}/testindex{test_index}_final_model_output_subspace{subspace_limit}_seed{args.seed}",
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

                masked_samples_dataset = TensorDataset(masked_samples_tensor)
                masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=64, shuffle=False)

                truthful_values = []
                for _data in masked_samples_data_loader:
                    truthful_values.append(torch.max(model(_data[0]).detach().cpu(), dim=1)[0])
                model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

                truthful_sample_masks = truthful_sample_masks.int()
                second_order_masks = []
                for each_mask in truthful_sample_masks:
                    tmp_mask = np.ones((each_mask.shape[0], each_mask.shape[0]))
                    each_mask_zero_index = torch.where(each_mask == 0)
                    for index in each_mask_zero_index[0]:
                        tmp_mask[index].fill(0)
                        tmp_mask[:, index].fill(0)
                    second_order_masks.append(tmp_mask)
                second_order_masks = np.stack(second_order_masks, axis=0)

                # flatten
                full_coeff = full_coeff.reshape(-1, 1)
                second_order_masks = second_order_masks.reshape(second_order_masks.shape[0], -1)

                shaptaylor_result = (
                        np.matmul(second_order_masks, full_coeff) + model(baseline)[0, predict].detach().cpu().numpy()).reshape(-1)

                answer = np.mean((model_truthful_values - shaptaylor_result) * sigmas)

                np.save(
                    f"{args.model}_shaptaylor/{test_index}/testindex{test_index}_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    shaptaylor_result)
                np.save(
                    f"{args.model}_shaptaylor/{test_index}/testindex{test_index}_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    model_truthful_values)
                np.save(
                    f"{args.model}_shaptaylor/{test_index}/testindex{test_index}_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    sigmas)
                np.save(
                    f"{args.model}_shaptaylor/{test_index}/testindex{test_index}_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    answer)

        torch.cuda.empty_cache()

elif args.algorithm == "faithshap":

    os.makedirs(f"{args.model}_faithshap", exist_ok=True)
    from torch_lr import TorchRidge
    from utils import revert_tensor_to_image, image_superpixel, generate_random_mask, \
        masks_to_transformed_tensor


    def faithshap_kernel(samples):
        samples = samples[np.where(samples.sum(axis=1) != samples.shape[1])]
        samples = samples[np.where(samples.sum(axis=1) != 0)]
        s = samples.sum(axis=1)
        M = np.ones(samples.shape[0]) * samples.shape[1]
        weights = (M - 1) / (comb(M, s) * s * (M - s))
        return weights, samples

    def faithshap_sampler(dim, shap_n_samples):
        inf_samples = np.concatenate([
            np.ones((1, dim)),
            np.zeros((1, dim)),
        ], axis=0)
        reg_samples = (np.random.rand(shap_n_samples - inf_samples.shape[0], dim) > .5)

        reg_weights, reg_samples = faithshap_kernel(reg_samples)
        reg_weights /= reg_weights.sum()
        reg_weights *= reg_weights.shape[0]
        inf_weights = np.ones(inf_samples.shape[0])
        inf_weights /= inf_weights.shape[0]
        inf_weights *= reg_weights.shape[0] * 3

        weights = np.concatenate([reg_weights, inf_weights], axis=0)
        samples = np.concatenate([reg_samples, inf_samples], axis=0).astype(np.bool)
        return torch.from_numpy(weights).to(device), \
               torch.from_numpy(samples).to(device).int()

    def calculate_second_order_faithshap_interaction(model, sample_tensor, baseline_tensor, dim, shap_n_samples, superpixel):

        weights, masks = faithshap_sampler(dim, shap_n_samples)

        samples_tensor = masks_to_transformed_tensor(masks, sample_tensor, superpixel).cpu()

        dataset = TensorDataset(samples_tensor)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        values = []
        for data in data_loader:
            values.append(torch.max(model(data[0].to(device)).detach().cpu(), dim=1)[0])
        values = torch.cat(values).squeeze().numpy()

        cross_terms = np.apply_along_axis(lambda s: (np.outer(s, s)).reshape(-1), 1, masks.cpu())
        term_mask = np.triu(np.ones((dim, dim), dtype=bool), k=0).reshape(-1)
        terms_to_keep = np.where(term_mask)
        cross_terms = cross_terms[:, terms_to_keep].squeeze()
        features = cross_terms.astype(bool)

        model = TorchRidge(alpha=.01, fit_intercept=True)
        weights = weights.cpu().numpy()
        model.fit(features, values, weights)

        full_coeff = np.zeros((dim * dim))
        full_coeff[terms_to_keep] = model.coef_[:cross_terms.shape[1]]
        full_coeff = full_coeff.reshape(dim, dim)

        return full_coeff

    # init variables
    C_range = [1, 2, 3]

    for test_index in pbar:
        os.makedirs(f"{args.model}_faithshap/{test_index}", exist_ok=True)

        data, label = test_data.__getitem__(test_index)  # data:tensor, label: int
        data = data.to(device).unsqueeze(0)  # create batch dim
        baseline = torch.zeros_like(data).to(device)

        output = model(data)
        predict = torch.argmax(output, axis=1)

        image = revert_tensor_to_image(data.squeeze(0))
        superpixel = image_superpixel(image, n_superpixels=args.n_superpixels)
        superpixel_one_hot = torch.nn.functional.one_hot(superpixel - torch.min(superpixel)).to(device)
        num_of_superpixels = superpixel.max().item()
        print("test_index", test_index, "num_of_superpixels", num_of_superpixels)

        shap_n_samples = min(1000, 2**num_of_superpixels)

        full_coeff = calculate_second_order_faithshap_interaction(model,
                                                                  data,
                                                                  baseline,
                                                                  num_of_superpixels,
                                                                  shap_n_samples,
                                                                  superpixel,
                                                                  )
        p_bar_info = ""
        for subspace_limit in [0, 1, 2, 4, 8]:
            args.subspace_limit = subspace_limit

            # calculate consistency
            n_samples = min(args.samples_min, 2**num_of_superpixels)
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



            truthful_sample_masks = truthful_sample_masks.int()
            second_order_masks = []
            for each_mask in truthful_sample_masks:
                tmp_mask = np.ones((each_mask.shape[0], each_mask.shape[0]))
                each_mask_zero_index = torch.where(each_mask == 0)
                for index in each_mask_zero_index[0]:
                    tmp_mask[index].fill(0)
                    tmp_mask[:, index].fill(0)
                second_order_masks.append(tmp_mask)
            second_order_masks = np.stack(second_order_masks, axis=0)

            # flatten
            full_coeff = full_coeff.reshape(-1, 1)
            second_order_masks = second_order_masks.reshape(second_order_masks.shape[0], -1)

            faithshap_result = (
                np.matmul(second_order_masks, full_coeff) + model(baseline)[0, predict].detach().cpu().numpy() ).reshape(-1)

            p_bar_info = p_bar_info + f"{subspace_limit} {np.mean(np.abs(faithshap_result - model_truthful_values))} "

            np.save(
                f"{args.model}_faithshap/{test_index}/testindex{test_index}_final_lasso_output_subspace{subspace_limit}_seed{args.seed}",
                faithshap_result)
            np.save(
                f"{args.model}_faithshap/{test_index}/testindex{test_index}_final_model_output_subspace{subspace_limit}_seed{args.seed}",
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

                masked_samples_dataset = TensorDataset(masked_samples_tensor)
                masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=64, shuffle=False)

                truthful_values = []
                for _data in masked_samples_data_loader:
                    truthful_values.append(torch.max(model(_data[0]).detach().cpu(), dim=1)[0])
                model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

                truthful_sample_masks = truthful_sample_masks.int()
                second_order_masks = []
                for each_mask in truthful_sample_masks:
                    tmp_mask = np.ones((each_mask.shape[0], each_mask.shape[0]))
                    each_mask_zero_index = torch.where(each_mask == 0)
                    for index in each_mask_zero_index[0]:
                        tmp_mask[index].fill(0)
                        tmp_mask[:, index].fill(0)
                    second_order_masks.append(tmp_mask)
                second_order_masks = np.stack(second_order_masks, axis=0)

                # flatten
                full_coeff = full_coeff.reshape(-1, 1)
                second_order_masks = second_order_masks.reshape(second_order_masks.shape[0], -1)

                faithshap_result = (
                        np.matmul(second_order_masks, full_coeff) + model(baseline)[0, predict].detach().cpu().numpy()).reshape(-1)

                answer = np.mean((model_truthful_values - faithshap_result) * sigmas)

                np.save(
                    f"{args.model}_faithshap/{test_index}/testindex{test_index}_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    faithshap_result)
                np.save(
                    f"{args.model}_faithshap/{test_index}/testindex{test_index}_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    model_truthful_values)
                np.save(
                    f"{args.model}_faithshap/{test_index}/testindex{test_index}_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    sigmas)
                np.save(
                    f"{args.model}_faithshap/{test_index}/testindex{test_index}_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    answer)

        torch.cuda.empty_cache()





