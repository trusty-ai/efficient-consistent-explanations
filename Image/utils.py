import copy
import torch

import numpy as np

from skimage.segmentation import slic
from torchvision import transforms
from PIL import Image
from itertools import product
from scipy.special import comb
from sklearn.preprocessing import PolynomialFeatures

transform_mean = [0.485, 0.456, 0.406]
transform_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=transform_mean,
        std=transform_std
    )
])

transform_little = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=transform_mean,
        std=transform_std
    )
])

def image_superpixel(transformed_image, n_superpixels):
    superpixels = torch.from_numpy(slic(
        transformed_image, n_segments=n_superpixels, compactness=20, sigma=3))

    return superpixels

def revert_tensor_to_image(transformed_tensor):
    transformed_tensor = copy.deepcopy(transformed_tensor)

    assert len(transformed_tensor.shape) == 3, f"{transformed_tensor.shape}"

    for t, m, s in zip(transformed_tensor, transform_mean, transform_std):
        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
    transformed_tensor = transformed_tensor * 255
    result = Image.fromarray(np.transpose(transformed_tensor.cpu().numpy(), (1, 2, 0)).astype(np.uint8))

    return result

def save_superpixel(image, superpixel):
    num_of_superpixels = superpixel.max().item()
    if superpixel.min() == 1:
        superpixel = superpixel - 1
    for i in range(num_of_superpixels):
        mask = torch.zeros(num_of_superpixels)
        mask[i] = 1

        result = mask_to_image_patch(mask, image, superpixel)  # 0 1 mask
        result.save(f'tmp{i}.png')

def mask_to_image_patch(mask, image, superpixel):
    assert mask.min()==0
    assert mask.max()==1
    if superpixel.min()==1:
        superpixel = superpixel - 1
    num_of_superpixels = superpixel.max().item()

    mask_indicator_dummy = (torch.arange(0, num_of_superpixels)[mask.to(torch.bool)]).unsqueeze(0).unsqueeze(0)
    mask = torch.any(mask_indicator_dummy == superpixel.unsqueeze(2), 2).unsqueeze(-1).detach().cpu().numpy()
    mask = mask.astype(np.int)

    image_partial = (np.array(image) * mask).astype(np.uint8)
    result = Image.fromarray(image_partial)

    return result

def masks_to_image_patch(masks, image, superpixel):
    assert masks.min()==0
    assert masks.max()==1
    num_of_superpixels = superpixel.max().item()
    if superpixel.min()==1:
        superpixel = superpixel - 1
    result = []
    for mask in masks:

        mask_indicator_dummy = (torch.arange(0, num_of_superpixels)[mask.to(torch.bool)]).unsqueeze(0).unsqueeze(0)
        mask = torch.any(mask_indicator_dummy == superpixel.unsqueeze(2), 2).unsqueeze(-1).detach().cpu().numpy()
        mask = mask.astype(np.int)

        image_partial = (np.array(image) * mask).astype(np.uint8)
        result.append(Image.fromarray(image_partial))

    return result

def masks_to_data_patch(masks, data, superpixel):
    assert masks.min() >= 0
    num_of_superpixels = superpixel.max().item()
    if superpixel.min() == 1:
        superpixel = superpixel - 1
    result = []
    # print("masks_to_data_patch num_of_superpixels", num_of_superpixels)
    # print("masks_to_data_patch superpixel", superpixel)
    for mask in masks:
        mask_indicator_dummy = (torch.arange(0, num_of_superpixels)[mask.to(torch.bool)]).unsqueeze(0).unsqueeze(0)
        mask = torch.any(mask_indicator_dummy == superpixel.unsqueeze(2), 2).unsqueeze(0).to(data.device)
        # print(mask.sum())

        image_partial = (data * mask)
        result.append(image_partial)

    return result

def masks_to_transformed_tensor(masks, data, superpixel):
    assert len(masks.shape)==2

    result = masks_to_data_patch(masks, data[0], superpixel)
    result = torch.stack(result, dim=0)

    return result


def generate_random_mask(args, x, n_samples=1000, length=0, subspace_limit=0):
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
            # print(column_index_every_row[0], column_index_every_row[1], column_index_every_row[2])

            mask_matrix = np.ones((n_samples, length))
            for _i in range(n_samples):
                mask_matrix[_i, column_index_every_row[_i]] = 0
            mask_matrix = mask_matrix * 2 - 1
            # assert 1==0
        else:
            mask_matrix = np.array(list(product([0, 1], repeat=length)))
            mask_matrix = mask_matrix[np.where(mask_matrix.sum(axis=1) >= length - subspace_limit)[0], :].squeeze()
            mask_matrix = mask_matrix * 2 - 1

    return mask_matrix

def group_attribution_mask(attribution, superpixel, take_average=False):
    assert attribution.shape[0] == 1
    attribution = attribution[0]
    num_of_superpixels = superpixel.max().item()
    if superpixel.min() == 1:
        superpixel = superpixel - 1
    result  = []
    for each_superpixel_index in range(num_of_superpixels):
        mask = torch.zeros(num_of_superpixels)
        mask[each_superpixel_index] = 1
        mask_indicator_dummy = (torch.arange(0, num_of_superpixels)[mask.to(torch.bool)]).unsqueeze(0).unsqueeze(0)
        mask = torch.any(mask_indicator_dummy == superpixel.unsqueeze(2), 2).unsqueeze(0).to(attribution.device)

        ans = attribution * mask

        if take_average:
            result.append(torch.sum(ans) / torch.sum(mask) / (attribution.shape[0] / mask.shape[0]))
        else:
            result.append(torch.sum(ans))

    result = torch.stack(result, dim=0)
    result = result.reshape(-1, 1)
    return result

def expand_basis_fun(basis, degree):
    one_test_sample_length = basis.shape[1]
    n_samples = basis.shape[0]
    if degree == 2:
        second_order_terms = np.matmul(basis[:, :, np.newaxis], basis[:, np.newaxis, :]).reshape(basis.shape[0], -1)
        terms_to_keep = \
        np.where(np.triu(np.ones((one_test_sample_length, one_test_sample_length)), k=1).reshape(-1) == 1)[0]
        second_order_terms = second_order_terms[:, terms_to_keep]
        expanded_basis = np.hstack([np.ones(n_samples).reshape(-1, 1), basis, second_order_terms])
    else:
        basis_extender = PolynomialFeatures(degree, interaction_only=True)
        expanded_basis = basis_extender.fit_transform(np.array(basis)).tolist()
        expanded_basis = np.array(expanded_basis)

    return expanded_basis