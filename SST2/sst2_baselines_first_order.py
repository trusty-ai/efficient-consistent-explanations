import os
import torch
import spacy
import random
import argparse
import itertools

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from itertools import product
from torchtext.legacy import data
from sst2_cnn_model import CNN, CNN_truncate
from scipy.special import comb
from utils import train, evaluate, count_parameters, epoch_time, expand_basis_fun, paragraph_to_sentence

from captum.attr import LimeBase


parser = argparse.ArgumentParser(description='consistent args')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--long_sentence_trucate', type=int, default=50, help='trucate size')
parser.add_argument('--modelpath', type=str, default="model_save/tut4-model-epoch4.pt", help='model path')
parser.add_argument('--subspace_limit', type=int, default=0, help='subspace_limit')
parser.add_argument('--samples_min', type=int, default=2000, help='samples_min')

parser.add_argument('--algorithm', type=str, default="lime", help='algorithm name')

parser.add_argument('--split_start', type=int, default=0, help='accelerate', required=False)
parser.add_argument('--split_end', type=int, default=0, help='accelerate', required=False)


args = parser.parse_args()

assert args.algorithm in ["lime", "ig", "ks"]

SEED = args.seed

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm',
                  batch_first = True)
LABEL = data.LabelField(dtype = torch.float)

fields = {'sentence': ('text', TEXT), 'label': ('label', LABEL)}
train_data, test_data=data.TabularDataset.splits(path='.',
                                                 train='sst2data/train.tsv',
                                                 test='sst2data/dev.tsv',
                                                 format='tsv',
                                                 fields=fields)

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data,
                 max_size=MAX_VOCAB_SIZE,
                 vectors="glove.6B.100d",
                 unk_init=torch.Tensor.normal_)

LABEL.build_vocab({'0': 0, '1': 1})

BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]


if args.long_sentence_trucate == 0:
    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
else:
    model = CNN_truncate(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX, args.long_sentence_trucate)

print(f'The model has {count_parameters(model):,} trainable parameters')

criterion = nn.BCELoss()

model = model.to(device)
criterion = criterion.to(device)

model.load_state_dict(torch.load(args.modelpath))

test_loss, test_acc = evaluate(args, model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')





nlp = spacy.load('en_core_web_sm')

pbar = tqdm(range((len(test_data))))

if args.split_end - args.split_start == 0 and args.split_start == 0:
    pbar = tqdm(range((len(test_data))))
else:
    assert args.algorithm in ["ig", "lime", "ks"], "Other algorithms do not support acceleration"
    pbar = tqdm(range(args.split_start, args.split_end))


if args.algorithm == "lime":
    def forward_func(text):
        out = model(text)
        return out


    def exp_embedding_cosine_distance(original_inp, perturbed_inp, _, **kwargs):
        original_emb = model.embedding(original_inp)
        perturbed_emb = model.embedding(perturbed_inp)
        original_emb = torch.mean(original_emb, dim=1)
        perturbed_emb = torch.mean(perturbed_emb, dim=1)
        distance = 1 - F.cosine_similarity(original_emb, perturbed_emb, dim=1)

        return torch.exp(-1 * (distance ** 2) / 2)


    def bernoulli_perturb(text, **kwargs):
        binary_vec_sum = 0
        while binary_vec_sum == 0:
            probs = torch.ones_like(text) * 0.5
            binary_vec = torch.bernoulli(probs).long()
            binary_vec_sum = torch.sum(binary_vec)
        return binary_vec


    def interp_to_input(interp_sample, original_input, **kwargs):
        return original_input[interp_sample.bool()].view(original_input.size(0), -1)


    def text_list_to_token_tensor(tokenized, length):
        if len(tokenized) < length:
            tokenized += ['<pad>'] * (length - len(tokenized))
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(0)
        return tensor


    def generate_random_mask(args, x, n_samples=1000, subspace_limit=0):
        # should return 1 and -1
        assert x.shape[0] == 1  # default: batch size
        length = x.shape[1]
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

        return mask_matrix


    def mask_to_masked_sample(masks_tensor, sample_tensor, pad_idx=1):
        sentence_length = sample_tensor.shape[1]
        return_tensor = []
        for each_mask in masks_tensor:
            _tmp = torch.masked_select(sample_tensor, each_mask)
            _tmp = F.pad(_tmp, (0, sentence_length - torch.sum(each_mask)), "constant", pad_idx)
            return_tensor.append(_tmp)

        return_tensor = torch.vstack(return_tensor)
        return return_tensor

    from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso

    lasso_lime_base = LimeBase(
        forward_func,
        interpretable_model=SkLearnLasso(alpha=0.001),
        similarity_func=exp_embedding_cosine_distance,
        perturb_func=bernoulli_perturb,
        perturb_interpretable_space=True,
        from_interp_rep_transform=interp_to_input,
        to_interp_rep_transform=None
    )

    final_lime_output_0 = []
    final_model_output_0 = []
    final_lime_output_1 = []
    final_model_output_1 = []
    final_lime_output_2 = []
    final_model_output_2 = []
    final_lime_output_4 = []
    final_model_output_4 = []
    final_lime_output_8 = []
    final_model_output_8 = []
    final_lime_output_16 = []
    final_model_output_16 = []
    final_lime_output_32 = []
    final_model_output_32 = []

    # init variables
    C_range = [1, 2, 3]
    for C in C_range:
        for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
            exec(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit} = []")
            exec(f"final_truthful_model_C_{C}_subspace_{subspace_limit} = []")
            exec(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit} = []")
            exec(f"final_truthful_answer_C_{C}_subspace_{subspace_limit} = []")

    for test_index in pbar:
        one_test_data = test_data.__getitem__(test_index)
        if args.long_sentence_trucate != 0:
            one_test_sample = one_test_data.text[0:args.long_sentence_trucate]  # list of string words
        else:
            one_test_sample = one_test_data.text

        one_test_sample_length = len(one_test_sample)
        sample_tensor = text_list_to_token_tensor(one_test_sample, length=len(one_test_sample))
        lime_coeff = lasso_lime_base.attribute(
            sample_tensor,  # add batch dimension for Captum
            n_samples=min(args.samples_min, 2 ** sample_tensor.shape[1]),
            show_progress=False
        )  # same shape with sample_tensor

        p_bar_info = ""
        for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
            args.subspace_limit = subspace_limit

            n_samples = min(args.samples_min, 2 ** sample_tensor.shape[1])
            truthful_sample_basis = generate_random_mask(args,
                                                         text_list_to_token_tensor(one_test_sample,
                                                                                   length=len(one_test_sample)),
                                                         n_samples=n_samples,
                                                         subspace_limit=args.subspace_limit)  # 1s and -1s
            truthful_sample_masks = torch.from_numpy((truthful_sample_basis + 1) / 2).cuda().bool()

            # process model f output
            masked_samples_tensor = mask_to_masked_sample(truthful_sample_masks, sample_tensor,
                                                          pad_idx=PAD_IDX)
            masked_samples_tensor = masked_samples_tensor.long()

            masked_samples_dataset = TensorDataset(masked_samples_tensor)
            masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=512, shuffle=False)

            truthful_values = []
            for _data in masked_samples_data_loader:
                truthful_values.append(model(_data[0]).detach().cpu())
            model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

            lime_result = (
                    np.matmul(truthful_sample_masks.cpu().numpy(), lime_coeff.reshape(-1, 1).cpu().numpy()) + lasso_lime_base.interpretable_model.bias().cpu().numpy()).reshape(-1)

            p_bar_info = p_bar_info + f"{subspace_limit} {np.mean(np.abs(lime_result - model_truthful_values))} "

            eval(f"final_lime_output_{subspace_limit}").append(lime_result)
            eval(f"final_model_output_{subspace_limit}").append(model_truthful_values)
        pbar.set_description("sentence length: %d" % (sample_tensor.shape[1],))

        # loop for C
        for C in C_range:
            x = np.arange(one_test_sample_length)
            combination_mask_index_under_degree = []
            truthful_degree = np.arange(0, C + 1).astype(int)
            for each_limit in truthful_degree:
                combination_mask_index_under_degree += list(itertools.combinations(x, each_limit))
            combination_mask = np.ones((len(combination_mask_index_under_degree), one_test_sample_length))
            for i, each_mask in enumerate(combination_mask_index_under_degree):
                for index in each_mask:
                    combination_mask[i][index] = 0

            combination_mask = (1 - combination_mask)  # number of 1s <= 3, this is kai_s

            # loop for subspace
            for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
                args.subspace_limit = subspace_limit
                truthful_sample_basis = generate_random_mask(args,
                                                             text_list_to_token_tensor(one_test_sample,
                                                                                       length=len(one_test_sample)),
                                                             n_samples=n_samples,
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
                masked_samples_tensor = mask_to_masked_sample(truthful_sample_masks, sample_tensor,
                                                              pad_idx=PAD_IDX)
                masked_samples_tensor = masked_samples_tensor.long()

                masked_samples_dataset = TensorDataset(masked_samples_tensor)
                masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=512, shuffle=False)

                truthful_values = []
                for _data in masked_samples_data_loader:
                    truthful_values.append(model(_data[0]).detach().cpu())
                model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

                lime_result = (
                        np.matmul(truthful_sample_masks.cpu().numpy(), lime_coeff.reshape(-1,
                                                                                          1).cpu().numpy()) + lasso_lime_base.interpretable_model.bias().cpu().numpy()).reshape(
                    -1)

                answer = np.mean((model_truthful_values - lime_result) * sigmas)

                eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}").append(lime_result)
                eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}").append(model_truthful_values)
                eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}").append(sigmas)
                eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}").append(answer)


    os.makedirs("lime", exist_ok=True)
    for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
        if args.split_end - args.split_start == 0 and args.split_start == 0:
            np.save(f"lime/lime_final_lasso_output_subspace{subspace_limit}_seed{args.seed}",
                    eval(f"final_lime_output_{subspace_limit}"))
            np.save(f"lime/lime_final_model_output_subspace{subspace_limit}_seed{args.seed}",
                    eval(f"final_model_output_{subspace_limit}"))
        else:
            np.save(
                f"lime/lime_final_lasso_output_subspace{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                eval(f"final_lime_output_{subspace_limit}"))
            np.save(
                f"lime/lime_final_model_output_subspace{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                eval(f"final_model_output_{subspace_limit}"))
    for C in C_range:
        for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
            if args.split_end - args.split_start == 0 and args.split_start == 0:
                np.save(
                    f"lime/lime_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"lime/lime_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"lime/lime_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"lime/lime_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}"))
            else:
                np.save(
                    f"lime/lime_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"lime/lime_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"lime/lime_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"lime/lime_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}"))

elif args.algorithm == "ig":
    from captum.attr import LayerIntegratedGradients, TokenReferenceBase

    token_reference = TokenReferenceBase(reference_token_idx=PAD_IDX)
    lig = LayerIntegratedGradients(model, model.embedding)

    def text_list_to_token_tensor(tokenized, length):
        if len(tokenized) < length:
            tokenized += ['<pad>'] * (length - len(tokenized))
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(0)
        return tensor

    def generate_random_mask(args, x, n_samples=1000, subspace_limit=0):
        # should return 1 and -1
        assert x.shape[0] == 1  # default: batch size
        length = x.shape[1]
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

        return mask_matrix

    def mask_to_masked_sample(masks_tensor, sample_tensor, pad_idx=1):
        sentence_length = sample_tensor.shape[1]
        return_tensor = []
        for each_mask in masks_tensor:
            _tmp = torch.masked_select(sample_tensor, each_mask)
            _tmp = F.pad(_tmp, (0, sentence_length - torch.sum(each_mask)), "constant", pad_idx)
            return_tensor.append(_tmp)

        return_tensor = torch.vstack(return_tensor)
        return return_tensor

    def interpret_sentence(model, input_indices, reference_indices, min_len=5, label=0):
        # text: list of tokens
        model.zero_grad()

        # input_indices dim: [sequence_length]
        seq_length = input_indices.shape[1]

        # compute attributions and approximation delta using layer integrated gradients
        attributions_ig, delta = lig.attribute(input_indices, reference_indices, n_steps=500,
                                               return_convergence_delta=True)
        return attributions_ig, delta


    final_ig_output_0 = []
    final_model_output_0 = []
    final_ig_output_1 = []
    final_model_output_1 = []
    final_ig_output_2 = []
    final_model_output_2 = []
    final_ig_output_4 = []
    final_model_output_4 = []
    final_ig_output_8 = []
    final_model_output_8 = []
    final_ig_output_16 = []
    final_model_output_16 = []
    final_ig_output_32 = []
    final_model_output_32 = []

    # init variables
    C_range = [1, 2, 3]
    for C in C_range:
        for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
            exec(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit} = []")
            exec(f"final_truthful_model_C_{C}_subspace_{subspace_limit} = []")
            exec(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit} = []")
            exec(f"final_truthful_answer_C_{C}_subspace_{subspace_limit} = []")

    for test_index in pbar:
        one_test_data = test_data.__getitem__(test_index)
        if args.long_sentence_trucate != 0:
            one_test_sample = one_test_data.text[0:args.long_sentence_trucate]  # list of string words
        else:
            one_test_sample = one_test_data.text

        sample_tensor = text_list_to_token_tensor(one_test_sample,
                                                  length=max(len(one_test_sample), 5))  # greater than kernel size
        one_test_sample_length = sample_tensor.shape[1]

        reference_indices = token_reference.generate_reference(sample_tensor.shape[1], device=device).unsqueeze(0)

        attributions_ig, delta = interpret_sentence(model, sample_tensor, reference_indices)
        attributions_ig = attributions_ig.sum(dim=2)[:, 0: sample_tensor.shape[1]]

        p_bar_info = ""
        for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:

            args.subspace_limit = subspace_limit

            n_samples = min(args.samples_min, 2 ** sample_tensor.shape[1])
            truthful_sample_basis = generate_random_mask(args,
                                                         text_list_to_token_tensor(one_test_sample,
                                                                                   length=len(one_test_sample)),
                                                         n_samples=n_samples,
                                                         subspace_limit=args.subspace_limit)  # 1s and -1s
            truthful_sample_masks = torch.from_numpy((truthful_sample_basis + 1) / 2).cuda().bool()

            # process model f output
            masked_samples_tensor = mask_to_masked_sample(truthful_sample_masks, sample_tensor,
                                                          pad_idx=PAD_IDX)
            masked_samples_tensor = masked_samples_tensor.long()

            masked_samples_dataset = TensorDataset(masked_samples_tensor)
            masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=512, shuffle=False)

            truthful_values = []
            for _data in masked_samples_data_loader:
                truthful_values.append(model(_data[0]).detach().cpu())
            model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

            ig_result = (
                    np.matmul(truthful_sample_masks.cpu().numpy(), attributions_ig.reshape(-1, 1).cpu().numpy()) - delta.cpu().numpy() + model(reference_indices).detach().cpu().numpy() ).reshape(-1)

            p_bar_info = p_bar_info + f"{subspace_limit} {np.mean(np.abs(ig_result - model_truthful_values))} "


            eval(f"final_ig_output_{args.subspace_limit}").append(ig_result)
            eval(f"final_model_output_{args.subspace_limit}").append(model_truthful_values)
        pbar.set_description("sentence length: %d" % (sample_tensor.shape[1],))

        # loop for C
        for C in C_range:
            x = np.arange(one_test_sample_length)
            combination_mask_index_under_degree = []
            truthful_degree = np.arange(0, C + 1).astype(int)
            for each_limit in truthful_degree:
                combination_mask_index_under_degree += list(itertools.combinations(x, each_limit))
            combination_mask = np.ones((len(combination_mask_index_under_degree), one_test_sample_length))
            for i, each_mask in enumerate(combination_mask_index_under_degree):
                for index in each_mask:
                    combination_mask[i][index] = 0

            combination_mask = (1 - combination_mask)  # number of 1s <= 3, this is kai_s

            # loop for subspace
            for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
                args.subspace_limit = subspace_limit
                truthful_sample_basis = generate_random_mask(args,
                                                             text_list_to_token_tensor(one_test_sample,
                                                                                       length=len(one_test_sample)),
                                                             n_samples=n_samples,
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
                masked_samples_tensor = mask_to_masked_sample(truthful_sample_masks, sample_tensor,
                                                              pad_idx=PAD_IDX)
                masked_samples_tensor = masked_samples_tensor.long()

                masked_samples_dataset = TensorDataset(masked_samples_tensor)
                masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=512, shuffle=False)

                truthful_values = []
                for _data in masked_samples_data_loader:
                    truthful_values.append(model(_data[0]).detach().cpu())
                model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

                ig_result = (
                        np.matmul(truthful_sample_masks.cpu().numpy(),
                                  attributions_ig.reshape(-1, 1).cpu().numpy()) - delta.cpu().numpy() + model(
                    reference_indices).detach().cpu().numpy()).reshape(-1)

                answer = np.mean((model_truthful_values - ig_result) * sigmas)

                eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}").append(ig_result)
                eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}").append(model_truthful_values)
                eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}").append(sigmas)
                eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}").append(answer)


    os.makedirs("ig", exist_ok=True)
    for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:

        if args.split_end - args.split_start == 0 and args.split_start == 0:
            np.save(f"ig/ig_final_lasso_output_subspace{subspace_limit}_seed{args.seed}",
                    eval(f"final_ig_output_{subspace_limit}"))
            np.save(f"ig/ig_final_model_output_subspace{subspace_limit}_seed{args.seed}",
                    eval(f"final_model_output_{subspace_limit}"))
        else:
            np.save(
                f"ig/ig_final_lasso_output_subspace{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                eval(f"final_ig_output_{subspace_limit}"))
            np.save(
                f"ig/ig_final_model_output_subspace{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                eval(f"final_model_output_{subspace_limit}"))
    for C in C_range:
        for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
            if args.split_end - args.split_start == 0 and args.split_start == 0:
                np.save(
                    f"ig/ig_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"ig/ig_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"ig/ig_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"ig/ig_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}"))
            else:
                np.save(
                    f"ig/ig_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"ig/ig_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"ig/ig_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"ig/ig_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}"))

elif args.algorithm == "ks":
    from captum.attr import TokenReferenceBase, KernelShap

    token_reference = TokenReferenceBase(reference_token_idx=PAD_IDX)
    ks = KernelShap(model)

    def text_list_to_token_tensor(tokenized, length):
        if len(tokenized) < length:
            tokenized += ['<pad>'] * (length - len(tokenized))
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(0)
        return tensor

    def generate_random_mask(args, x, n_samples=1000, subspace_limit=0):
        # should return 1 and -1
        assert x.shape[0] == 1  # default: batch size
        length = x.shape[1]
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

        return mask_matrix

    def mask_to_masked_sample(masks_tensor, sample_tensor, pad_idx=1):
        sentence_length = sample_tensor.shape[1]
        return_tensor = []
        for each_mask in masks_tensor:
            _tmp = torch.masked_select(sample_tensor, each_mask)
            _tmp = F.pad(_tmp, (0, sentence_length - torch.sum(each_mask)), "constant", pad_idx)
            return_tensor.append(_tmp)

        return_tensor = torch.vstack(return_tensor)
        return return_tensor

    final_ks_output_0 = []
    final_model_output_0 = []
    final_ks_output_1 = []
    final_model_output_1 = []
    final_ks_output_2 = []
    final_model_output_2 = []
    final_ks_output_4 = []
    final_model_output_4 = []
    final_ks_output_8 = []
    final_model_output_8 = []
    final_ks_output_16 = []
    final_model_output_16 = []
    final_ks_output_32 = []
    final_model_output_32 = []

    # init variables
    C_range = [1, 2, 3]
    for C in C_range:
        for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
            exec(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit} = []")
            exec(f"final_truthful_model_C_{C}_subspace_{subspace_limit} = []")
            exec(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit} = []")
            exec(f"final_truthful_answer_C_{C}_subspace_{subspace_limit} = []")

    for test_index in pbar:
        one_test_data = test_data.__getitem__(test_index)
        if args.long_sentence_trucate != 0:
            one_test_sample = one_test_data.text[0:args.long_sentence_trucate]  # list of string words
        else:
            one_test_sample = one_test_data.text
        sample_tensor = text_list_to_token_tensor(one_test_sample,
                                                  length=max(len(one_test_sample), 5))  # greater than kernel size
        one_test_sample_length = sample_tensor.shape[1]

        reference_indices = token_reference.generate_reference(sample_tensor.shape[1], device=device).unsqueeze(0)

        attr = ks.attribute(inputs=sample_tensor,
                            baselines=reference_indices,
                            target=None,
                            n_samples=min(args.samples_min, 2 ** sample_tensor.shape[1]))

        p_bar_info = ""
        for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:

            args.subspace_limit = subspace_limit

            n_samples = min(args.samples_min, 2 ** sample_tensor.shape[1])
            truthful_sample_basis = generate_random_mask(args,
                                                         text_list_to_token_tensor(one_test_sample,
                                                                                   length=len(one_test_sample)),
                                                         n_samples=n_samples,
                                                         subspace_limit=args.subspace_limit)  # 1s and -1s
            truthful_sample_masks = torch.from_numpy((truthful_sample_basis + 1) / 2).cuda().bool()

            # process model f output
            masked_samples_tensor = mask_to_masked_sample(truthful_sample_masks, sample_tensor,
                                                          pad_idx=PAD_IDX)
            masked_samples_tensor = masked_samples_tensor.long()

            masked_samples_dataset = TensorDataset(masked_samples_tensor)
            masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=512, shuffle=False)

            truthful_values = []
            for _data in masked_samples_data_loader:
                truthful_values.append(model(_data[0]).detach().cpu())
            model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

            ks_result = (
                    np.matmul(truthful_sample_masks.cpu().numpy(),
                              attr.reshape(-1, 1).cpu().numpy()) + model(reference_indices).detach().cpu().numpy()  ).reshape(-1)

            p_bar_info = p_bar_info + f"{subspace_limit} {np.mean(np.abs(ks_result - model_truthful_values))} "

            eval(f"final_ks_output_{args.subspace_limit}").append(ks_result)
            eval(f"final_model_output_{args.subspace_limit}").append(model_truthful_values)
        pbar.set_description("sentence length: %d" % (sample_tensor.shape[1],))

        # loop for C
        for C in C_range:
            x = np.arange(one_test_sample_length)
            combination_mask_index_under_degree = []
            truthful_degree = np.arange(0, C + 1).astype(int)
            for each_limit in truthful_degree:
                combination_mask_index_under_degree += list(itertools.combinations(x, each_limit))
            combination_mask = np.ones((len(combination_mask_index_under_degree), one_test_sample_length))
            for i, each_mask in enumerate(combination_mask_index_under_degree):
                for index in each_mask:
                    combination_mask[i][index] = 0

            combination_mask = (1 - combination_mask)  # number of 1s <= 3, this is kai_s

            # loop for subspace
            for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
                args.subspace_limit = subspace_limit
                truthful_sample_basis = generate_random_mask(args,
                                                             text_list_to_token_tensor(one_test_sample,
                                                                                       length=len(one_test_sample)),
                                                             n_samples=n_samples,
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
                masked_samples_tensor = mask_to_masked_sample(truthful_sample_masks, sample_tensor,
                                                              pad_idx=PAD_IDX)
                masked_samples_tensor = masked_samples_tensor.long()

                masked_samples_dataset = TensorDataset(masked_samples_tensor)
                masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=512, shuffle=False)

                truthful_values = []
                for _data in masked_samples_data_loader:
                    truthful_values.append(model(_data[0]).detach().cpu())
                model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

                ks_result = (
                        np.matmul(truthful_sample_masks.cpu().numpy(),
                                  attr.reshape(-1, 1).cpu().numpy()) + model(
                    reference_indices).detach().cpu().numpy()).reshape(
                    -1)

                answer = np.mean((model_truthful_values - ks_result) * sigmas)

                eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}").append(ks_result)
                eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}").append(model_truthful_values)
                eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}").append(sigmas)
                eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}").append(answer)

    os.makedirs("ks", exist_ok=True)
    for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:

        if args.split_end - args.split_start == 0 and args.split_start == 0:
            np.save(f"ks/ks_final_lasso_output_subspace{subspace_limit}_seed{args.seed}",
                    eval(f"final_ks_output_{subspace_limit}"))
            np.save(f"ks/ks_final_model_output_subspace{subspace_limit}_seed{args.seed}",
                    eval(f"final_model_output_{subspace_limit}"))
        else:
            np.save(
                f"ks/ks_final_lasso_output_subspace{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                eval(f"final_ks_output_{subspace_limit}"))
            np.save(
                f"ks/ks_final_model_output_subspace{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                eval(f"final_model_output_{subspace_limit}"))
    for C in C_range:
        for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
            if args.split_end - args.split_start == 0 and args.split_start == 0:
                np.save(
                    f"ks/ks_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"ks/ks_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"ks/ks_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"ks/ks_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}"))
            else:
                np.save(
                    f"ks/ks_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"ks/ks_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"ks/ks_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"ks/ks_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}"))