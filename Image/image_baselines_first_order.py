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

from captum.attr import IntegratedGradients

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--datapath', metavar='DIR', nargs='?', default='/mnt/sda1/user/ImageNet/val/',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--split_start', type=int, default=0, help='accelerate', required=False)
parser.add_argument('--split_end', type=int, default=0, help='accelerate', required=False)
parser.add_argument('--subspace_limit', type=int, default=0, help='subspace_limit')
parser.add_argument('--samples_min', type=int, default=1000, help='samples_min')
parser.add_argument('--model', type=str, default='resnet18', help='model')

parser.add_argument('--algorithm', type=str, default="lime", help='algorithm name')
parser.add_argument('--n_superpixels', type=int, default=6, help='n_superpixels')

args = parser.parse_args()

assert args.algorithm in ["lime", "ig", "ks"]

SEED = args.seed

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils import transform

test_data = torchvision.datasets.ImageFolder(args.datapath, transform=transform)

random_index = np.arange(len(test_data))
np.random.shuffle(random_index)

if args.model == "resnet18":
    model = models.resnet18(pretrained=True)
elif args.model == "resnet34":
    model = models.resnet34(pretrained=True)
elif args.model == "resnet50":
    model = models.resnet50(pretrained=True)
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

pbar = tqdm(random_index)

if args.split_end - args.split_start == 0 and args.split_start == 0:
    pbar = tqdm(random_index)
else:
    assert args.algorithm in ["ig", "lime", "ks"], "Other algorithms do not support acceleration"
    pbar = tqdm(random_index[args.split_start: args.split_end])

if args.algorithm == "lime":
    from captum.attr import LimeBase
    from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
    os.makedirs(f"{args.model}_lime", exist_ok=True)
    from utils import revert_tensor_to_image, image_superpixel, save_superpixel, generate_random_mask, \
        masks_to_transformed_tensor, group_attribution_mask
    from captum.attr._core.lime import get_exp_kernel_similarity_function

    C_range = [1, 2, 3]
    for test_index in pbar:
        os.makedirs(f"{args.model}_lime/{test_index}", exist_ok=True)

        def forward_func(data):
            out = model(data)
            return out

        exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)

        def bernoulli_perturb(data, **kwargs):
            num_of_superpixels = kwargs["num_of_superpixels"]
            probs = torch.ones(num_of_superpixels) * 0.5
            binary_vec = torch.bernoulli(probs).reshape(1, -1).long()
            return binary_vec

        def interp_to_input(interp_sample, original_input, **kwargs):
            superpixel = kwargs["superpixel"]
            result = masks_to_transformed_tensor(interp_sample, original_input, superpixel)
            return result

        data, label = test_data.__getitem__(test_index)  # data:tensor, label: int
        data = data.to(device).unsqueeze(0)  # create batch dim
        baseline = torch.zeros_like(data).to(device)

        output = model(data)
        predict = torch.argmax(output, axis=1)

        image = revert_tensor_to_image(data.squeeze(0))
        superpixel = image_superpixel(image, n_superpixels=args.n_superpixels)
        num_of_superpixels = superpixel.max().item()

        n_samples = min(args.samples_min, 2 ** num_of_superpixels)

        lasso_lime_base = LimeBase(
            forward_func,
            interpretable_model=SkLearnLasso(alpha=0.001),
            similarity_func=exp_eucl_distance,
            perturb_func=bernoulli_perturb,
            perturb_interpretable_space=True,
            from_interp_rep_transform=interp_to_input,
            to_interp_rep_transform=None
        )
        lime_coeff = lasso_lime_base.attribute(
            data,  # add batch dimension for Captum
            target=predict.unsqueeze(0),
            n_samples=n_samples,
            perturbations_per_eval=4,
            show_progress=False,
            num_of_superpixels=num_of_superpixels,
            superpixel=superpixel
        )  # same shape with sample_tensor

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
            masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=2, shuffle=False)

            truthful_values = []
            for _data in masked_samples_data_loader:
                truthful_values.append(torch.max(model(_data[0]).detach().cpu(), dim=1)[0])
            model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

            lime_result = (
                    np.matmul(truthful_sample_masks.cpu().numpy(), lime_coeff.reshape(-1,
                                                                                      1).cpu().numpy()) + lasso_lime_base.interpretable_model.bias().cpu().numpy()).reshape(
                -1)

            p_bar_info = p_bar_info + f"{subspace_limit} {np.mean(np.abs(lime_result - model_truthful_values))} "

            np.save(
                f"{args.model}_lime/{test_index}/testindex{test_index}_final_lasso_output_subspace{subspace_limit}_seed{args.seed}",
                lime_result)
            np.save(
                f"{args.model}_lime/{test_index}/testindex{test_index}_final_model_output_subspace{subspace_limit}_seed{args.seed}",
                model_truthful_values)

        pbar.set_description("superpixel number: %d" % (num_of_superpixels))

        del truthful_sample_basis, truthful_sample_masks, lime_result, model_truthful_values, masked_samples_dataset, masked_samples_tensor, masked_samples_data_loader, truthful_values, output

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
                masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=2, shuffle=False)

                truthful_values = []
                for _data in masked_samples_data_loader:
                    truthful_values.append(torch.max(model(_data[0]).detach().cpu(), dim=1)[0])
                model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

                lime_result = (
                        np.matmul(truthful_sample_masks.cpu().numpy(), lime_coeff.reshape(-1,
                                                                                          1).cpu().numpy()) + lasso_lime_base.interpretable_model.bias().cpu().numpy()).reshape(
                    -1)

                answer = np.mean((model_truthful_values - lime_result) * sigmas)

                np.save(
                    f"{args.model}_lime/{test_index}/testindex{test_index}_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    lime_result)
                np.save(
                    f"{args.model}_lime/{test_index}/testindex{test_index}_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    model_truthful_values)
                np.save(
                    f"{args.model}_lime/{test_index}/testindex{test_index}_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    sigmas)
                np.save(
                    f"{args.model}_lime/{test_index}/testindex{test_index}_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    answer)

        del lasso_lime_base
        torch.cuda.empty_cache()

elif args.algorithm == "ig":
    os.makedirs(f"{args.model}_ig", exist_ok=True)
    from utils import revert_tensor_to_image, image_superpixel, save_superpixel, generate_random_mask, \
        masks_to_transformed_tensor, group_attribution_mask

    C_range = [1, 2, 3]
    for test_index in pbar:
        os.makedirs(f"{args.model}_ig/{test_index}", exist_ok=True)
        integrated_gradients = IntegratedGradients(model)

        data, label = test_data.__getitem__(test_index)  # data:tensor, label: int
        data = data.to(device).unsqueeze(0)  # create batch dim
        baseline = torch.zeros_like(data).to(device)

        output = model(data)
        predict = torch.argmax(output, axis=1)

        image = revert_tensor_to_image(data.squeeze(0))
        superpixel = image_superpixel(image, n_superpixels=args.n_superpixels)
        num_of_superpixels = superpixel.max().item()

        attributions_ig, delta = integrated_gradients.attribute(data, baselines=baseline, target=predict, n_steps=200,
                                                                return_convergence_delta=True, internal_batch_size=16)
        attributions_ig = group_attribution_mask(attributions_ig, superpixel)

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
            masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=16, shuffle=False)

            truthful_values = []
            for _data in masked_samples_data_loader:
                truthful_values.append(torch.max(model(_data[0]).detach().cpu(), dim=1)[0])
            model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

            ig_result = (
                    np.matmul(truthful_sample_masks.cpu().numpy(),
                              attributions_ig.reshape(-1, 1).cpu().numpy()) - delta.cpu().numpy() + model(
                baseline)[0, predict].detach().cpu().numpy()).reshape(-1)

            p_bar_info = p_bar_info + f"{subspace_limit} {np.mean(np.abs(ig_result - model_truthful_values))} "

            np.save(f"{args.model}_ig/{test_index}/testindex{test_index}_final_lasso_output_subspace{subspace_limit}_seed{args.seed}",
                    ig_result)
            np.save(f"{args.model}_ig/{test_index}/testindex{test_index}_final_model_output_subspace{subspace_limit}_seed{args.seed}",
                    model_truthful_values)

        pbar.set_description("superpixel number: %d" % (num_of_superpixels))

        del truthful_sample_basis, truthful_sample_masks, ig_result, model_truthful_values, masked_samples_dataset, masked_samples_tensor, masked_samples_data_loader, truthful_values, output

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
                masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=16, shuffle=False)

                truthful_values = []
                for _data in masked_samples_data_loader:
                    truthful_values.append(torch.max(model(_data[0]).detach().cpu(), dim=1)[0])
                model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

                ig_result = (
                        np.matmul(truthful_sample_masks.cpu().numpy(),
                                  attributions_ig.reshape(-1, 1).cpu().numpy()) - delta.cpu().numpy() + model(
                    baseline)[0, predict].detach().cpu().numpy()).reshape(-1)

                answer = np.mean((model_truthful_values - ig_result) * sigmas)

                np.save(
                    f"{args.model}_ig/{test_index}/testindex{test_index}_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    ig_result)
                np.save(
                    f"{args.model}_ig/{test_index}/testindex{test_index}_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    model_truthful_values)
                np.save(
                    f"{args.model}_ig/{test_index}/testindex{test_index}_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    sigmas)
                np.save(
                    f"{args.model}_ig/{test_index}/testindex{test_index}_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    answer)

        del attributions_ig, integrated_gradients
        torch.cuda.empty_cache()

elif args.algorithm == "ks":
    os.makedirs(f"{args.model}_ks", exist_ok=True)
    from utils import revert_tensor_to_image, image_superpixel, save_superpixel, generate_random_mask, \
        masks_to_transformed_tensor, group_attribution_mask
    from captum.attr import KernelShap

    C_range = [1, 2, 3]
    for test_index in pbar:
        os.makedirs(f"{args.model}_ks/{test_index}", exist_ok=True)
        ks = KernelShap(model)

        data, label = test_data.__getitem__(test_index)  # data:tensor, label: int
        data = data.to(device).unsqueeze(0)  # create batch dim
        baseline = torch.zeros_like(data).to(device)

        output = model(data)
        predict = torch.argmax(output, axis=1)

        image = revert_tensor_to_image(data.squeeze(0))
        superpixel = image_superpixel(image, n_superpixels=args.n_superpixels)
        num_of_superpixels = superpixel.max().item()

        attr = ks.attribute(inputs=data,
                            baselines=torch.zeros_like(data).to(device),
                            target=predict.unsqueeze(0),
                            feature_mask=superpixel.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device),
                            n_samples=min(args.samples_min, 2 ** num_of_superpixels))
        attr = group_attribution_mask(attr, superpixel, take_average=True)

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
            masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=16, shuffle=False)

            truthful_values = []
            for _data in masked_samples_data_loader:
                truthful_values.append(torch.max(model(_data[0]).detach().cpu(), dim=1)[0])
            model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

            ks_result = (
                    np.matmul(truthful_sample_masks.cpu().numpy(),
                              attr.reshape(-1, 1).cpu().numpy()) + model(
                baseline)[0, predict].detach().cpu().numpy()).reshape(-1)

            p_bar_info = p_bar_info + f"{subspace_limit} {np.mean(np.abs(ks_result - model_truthful_values))} "

            np.save(
                f"{args.model}_ks/{test_index}/testindex{test_index}_final_lasso_output_subspace{subspace_limit}_seed{args.seed}",
                ks_result)
            np.save(
                f"{args.model}_ks/{test_index}/testindex{test_index}_final_model_output_subspace{subspace_limit}_seed{args.seed}",
                model_truthful_values)

        pbar.set_description("superpixel number: %d" % (num_of_superpixels))

        del truthful_sample_basis, truthful_sample_masks, ks_result, model_truthful_values, masked_samples_dataset, masked_samples_tensor, masked_samples_data_loader, truthful_values, output

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
                masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=16, shuffle=False)

                truthful_values = []
                for _data in masked_samples_data_loader:
                    truthful_values.append(torch.max(model(_data[0]).detach().cpu(), dim=1)[0])
                model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

                ks_result = (
                        np.matmul(truthful_sample_masks.cpu().numpy(),
                                  attr.reshape(-1, 1).cpu().numpy()) + model(
                    baseline)[0, predict].detach().cpu().numpy()).reshape(-1)

                answer = np.mean((model_truthful_values - ks_result) * sigmas)

                np.save(
                    f"{args.model}_ks/{test_index}/testindex{test_index}_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    ks_result)
                np.save(
                    f"{args.model}_ks/{test_index}/testindex{test_index}_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    model_truthful_values)
                np.save(
                    f"{args.model}_ks/{test_index}/testindex{test_index}_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    sigmas)
                np.save(
                    f"{args.model}_ks/{test_index}/testindex{test_index}_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    answer)

        del ks
        torch.cuda.empty_cache()


