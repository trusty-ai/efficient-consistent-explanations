import os
import torch
import spacy
import random
import argparse
import itertools

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from scipy.special import comb
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from itertools import product
from torchtext.legacy import data
from sst2_cnn_model import CNN, CNN_truncate, CNN_truncate_head, CNN_truncate_tail
from utils import train, evaluate, count_parameters, epoch_time, expand_basis_fun, paragraph_to_sentence

from torch.autograd.functional import hessian
from torch.autograd.functional import jacobian

parser = argparse.ArgumentParser(description='consistent args')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--long_sentence_trucate', type=int, default=50, help='trucate size')
parser.add_argument('--modelpath', type=str, default="model_save/tut4-model-epoch4.pt", help='model path')
parser.add_argument('--subspace_limit', type=int, default=0, help='subspace_limit')
parser.add_argument('--samples_min', type=int, default=2000, help='samples_min')

parser.add_argument('--algorithm', type=str, default="ih", help='algorithm name')

parser.add_argument('--split_start', type=int, default=0, help='accelerate', required=False)
parser.add_argument('--split_end', type=int, default=0, help='accelerate', required=False)


args = parser.parse_args()

assert args.algorithm in ["ih", "shaptaylor", "faithshap"]

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
    assert args.algorithm in ["ih", "shaptaylor", "faithshap"], "Other algorithms do not support acceleration"
    pbar = tqdm(range(args.split_start, args.split_end))


if args.algorithm == "ih":
    print("start ih")

    from captum.attr import TokenReferenceBase

    model_token_to_embedding = CNN_truncate_head(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT,
                                                 PAD_IDX, args.long_sentence_trucate)
    model_token_to_embedding.embedding.weight.data.copy_(model.embedding.weight.data)
    model_token_to_embedding = model_token_to_embedding.to(device)

    model_embedding_to_output = CNN_truncate_tail(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM,
                                                  DROPOUT, PAD_IDX, args.long_sentence_trucate)
    model_embedding_to_output = model_embedding_to_output.to(device)
    model_embedding_to_output.load_state_dict(model.state_dict(), strict=False)

    model_token_to_embedding.eval()
    model_embedding_to_output.eval()

    token_reference = TokenReferenceBase(reference_token_idx=PAD_IDX)


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
                mask_matrix = mask_matrix.reshape(-1, length)
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

    final_ih_output_0 = []
    final_model_output_0 = []
    final_ih_output_1 = []
    final_model_output_1 = []
    final_ih_output_2 = []
    final_model_output_2 = []
    final_ih_output_4 = []
    final_model_output_4 = []
    final_ih_output_8 = []
    final_model_output_8 = []
    final_ih_output_16 = []
    final_model_output_16 = []
    final_ih_output_32 = []
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

        reference_indices = token_reference.generate_reference(sample_tensor.shape[1], device=device).unsqueeze(0)

        sample_embedding = model_token_to_embedding(sample_tensor)
        reference_embedding = model_token_to_embedding(reference_indices)

        def func(blended_mask):
            mask_q = blended_mask.reshape(1, 1, -1, 1).repeat(1, 1, 1, EMBEDDING_DIM)
            return model_embedding_to_output(sample_embedding * mask_q + reference_embedding * (1 - mask_q))

        acc_hessian, acc_grad = integrated_hessians(func, sample_tensor.shape[1],)

        acc_hessian = acc_hessian.reshape(-1, 1).cpu().numpy()
        acc_grad = acc_grad.reshape(-1, 1).cpu().numpy()
        incompleteness = model(sample_tensor) - model(reference_indices) - acc_hessian.sum() - acc_grad.sum()
        incompleteness = incompleteness.detach().cpu().numpy()

        p_bar_info = ""
        for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:

            args.subspace_limit = subspace_limit

            # calculate consistency
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
                    + model(reference_indices).detach().cpu().numpy()
                    + incompleteness
            ).reshape(-1)

            p_bar_info = p_bar_info + f"{subspace_limit} {np.mean(np.abs(ih_result - model_truthful_values))} "

            eval(f"final_ih_output_{subspace_limit}").append(ih_result)
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
                        + model(reference_indices).detach().cpu().numpy()
                        + incompleteness
                ).reshape(-1)

                answer = np.mean((model_truthful_values - ih_result) * sigmas)

                eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}").append(ih_result)
                eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}").append(model_truthful_values)
                eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}").append(sigmas)
                eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}").append(answer)


    os.makedirs("ih", exist_ok=True)
    for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
        if args.split_end - args.split_start == 0 and args.split_start == 0:
            np.save(f"ih/ih_final_lasso_output_subspace{subspace_limit}_seed{args.seed}",
                    eval(f"final_ih_output_{subspace_limit}"))
            np.save(f"ih/ih_final_model_output_subspace{subspace_limit}_seed{args.seed}",
                    eval(f"final_model_output_{subspace_limit}"))
        else:
            np.save(
                f"ih/ih_final_lasso_output_subspace{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                eval(f"final_ih_output_{subspace_limit}"))
            np.save(
                f"ih/ih_final_model_output_subspace{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                eval(f"final_model_output_{subspace_limit}"))

    for C in C_range:
        for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
            if args.split_end - args.split_start == 0 and args.split_start == 0:
                np.save(
                    f"ih/ih_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"ih/ih_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"ih/ih_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"ih/ih_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}"))
            else:
                np.save(
                    f"ih/ih_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"ih/ih_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"ih/ih_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"ih/ih_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}"))


elif args.algorithm == "shaptaylor":

    from captum.attr import TokenReferenceBase
    from torch_lr import TorchRidge

    model_token_to_embedding = CNN_truncate_head(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT,
                                                 PAD_IDX, args.long_sentence_trucate)
    model_token_to_embedding.embedding.weight.data.copy_(model.embedding.weight.data)
    model_token_to_embedding = model_token_to_embedding.to(device)

    model_embedding_to_output = CNN_truncate_tail(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM,
                                                  DROPOUT, PAD_IDX, args.long_sentence_trucate)
    model_embedding_to_output = model_embedding_to_output.to(device)
    model_embedding_to_output.load_state_dict(model.state_dict(), strict=False)

    model_token_to_embedding.eval()
    model_embedding_to_output.eval()

    token_reference = TokenReferenceBase(reference_token_idx=PAD_IDX)


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
                mask_matrix = mask_matrix.reshape(-1, length)
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

    def calculate_second_order_shap_taylor_interaction(model, sample_tensor, reference_tensor, dim, shap_n_samples):

        weights, samples = shap_sampler(dim, shap_n_samples)
        samples_tokens = samples * sample_tensor + (1-samples) * reference_tensor
        samples_tokens = samples_tokens.long().cpu()

        dataset = TensorDataset(samples_tokens)
        data_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16)
        values = []
        for data in data_loader:
            data = data[0].cuda()
            values.append(model(data).detach().cpu())
        values = torch.cat(values).squeeze().numpy()

        cross_terms = np.apply_along_axis(lambda s: (np.outer(s, s)).reshape(-1), 1, samples.cpu())
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

    final_shaptaylor_output_0 = []
    final_model_output_0 = []
    final_shaptaylor_output_1 = []
    final_model_output_1 = []
    final_shaptaylor_output_2 = []
    final_model_output_2 = []
    final_shaptaylor_output_4 = []
    final_model_output_4 = []
    final_shaptaylor_output_8 = []
    final_model_output_8 = []
    final_shaptaylor_output_16 = []
    final_model_output_16 = []
    final_shaptaylor_output_32 = []
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

        reference_indices = token_reference.generate_reference(sample_tensor.shape[1], device=device).unsqueeze(0)

        sample_embedding = model_token_to_embedding(sample_tensor)
        reference_embedding = model_token_to_embedding(reference_indices)

        shap_n_samples = min(1000, 2**sample_tensor.shape[1])

        full_coeff = calculate_second_order_shap_taylor_interaction(model,
                                                                    sample_tensor,
                                                                    reference_indices,
                                                                    sample_tensor.shape[1],
                                                                    shap_n_samples,
                                                                    )
        p_bar_info = ""
        for subspace_limit in [0, 1, 2,4, 8, 16, 32]:
            args.subspace_limit = subspace_limit

            # calculate consistency
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
                np.matmul(second_order_masks, full_coeff) + model(reference_indices).detach().cpu().numpy() ).reshape(-1)

            p_bar_info = p_bar_info + f"{subspace_limit} {np.mean(np.abs(shaptaylor_result - model_truthful_values))} "

            eval(f"final_shaptaylor_output_{args.subspace_limit}").append(shaptaylor_result)
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
                        np.matmul(second_order_masks, full_coeff) + model(
                    reference_indices).detach().cpu().numpy()).reshape(-1)

                answer = np.mean((model_truthful_values - shaptaylor_result) * sigmas)

                eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}").append(shaptaylor_result)
                eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}").append(model_truthful_values)
                eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}").append(sigmas)
                eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}").append(answer)

    os.makedirs("shaptaylor", exist_ok=True)
    for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
        if args.split_end - args.split_start == 0 and args.split_start == 0:
            np.save(f"shaptaylor/shaptaylor_final_lasso_output_subspace{subspace_limit}_seed{args.seed}",
                    eval(f"final_shaptaylor_output_{subspace_limit}"))
            np.save(f"shaptaylor/shaptaylor_final_model_output_subspace{subspace_limit}_seed{args.seed}",
                    eval(f"final_model_output_{subspace_limit}"))
        else:
            np.save(
                f"shaptaylor/shaptaylor_final_lasso_output_subspace{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                eval(f"final_shaptaylor_output_{subspace_limit}"))
            np.save(
                f"shaptaylor/shaptaylor_final_model_output_subspace{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                eval(f"final_model_output_{subspace_limit}"))

    for C in C_range:
        for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
            if args.split_end - args.split_start == 0 and args.split_start == 0:
                np.save(
                    f"shaptaylor/shaptaylor_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"shaptaylor/shaptaylor_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"shaptaylor/shaptaylor_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"shaptaylor/shaptaylor_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}"))
            else:
                np.save(
                    f"shaptaylor/shaptaylor_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"shaptaylor/shaptaylor_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"shaptaylor/shaptaylor_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"shaptaylor/shaptaylor_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}"))

elif args.algorithm == "faithshap":

    from captum.attr import TokenReferenceBase
    from torch_lr import TorchRidge

    model_token_to_embedding = CNN_truncate_head(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT,
                                                 PAD_IDX, args.long_sentence_trucate)
    model_token_to_embedding.embedding.weight.data.copy_(model.embedding.weight.data)
    model_token_to_embedding = model_token_to_embedding.to(device)

    model_embedding_to_output = CNN_truncate_tail(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM,
                                                  DROPOUT, PAD_IDX, args.long_sentence_trucate)
    model_embedding_to_output = model_embedding_to_output.to(device)
    model_embedding_to_output.load_state_dict(model.state_dict(), strict=False)

    model_token_to_embedding.eval()
    model_embedding_to_output.eval()

    token_reference = TokenReferenceBase(reference_token_idx=PAD_IDX)


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
                mask_matrix = mask_matrix.reshape(-1, length)
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

    def calculate_second_order_faithshap_interaction(model, sample_tensor, reference_tensor, dim, shap_n_samples):

        weights, samples = faithshap_sampler(dim, shap_n_samples)
        samples_tokens = samples * sample_tensor + (1-samples) * reference_tensor
        samples_tokens = samples_tokens.long().cpu()

        dataset = TensorDataset(samples_tokens)
        data_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16)
        values = []
        for data in data_loader:
            data = data[0].cuda()
            values.append(model(data).detach().cpu())
        values = torch.cat(values).squeeze().numpy()

        cross_terms = np.apply_along_axis(lambda s: (np.outer(s, s)).reshape(-1), 1, samples.cpu())
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

    final_faithshap_output_0 = []
    final_model_output_0 = []
    final_faithshap_output_1 = []
    final_model_output_1 = []
    final_faithshap_output_2 = []
    final_model_output_2 = []
    final_faithshap_output_4 = []
    final_model_output_4 = []
    final_faithshap_output_8 = []
    final_model_output_8 = []
    final_faithshap_output_16 = []
    final_model_output_16 = []
    final_faithshap_output_32 = []
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

        reference_indices = token_reference.generate_reference(sample_tensor.shape[1], device=device).unsqueeze(0)

        sample_embedding = model_token_to_embedding(sample_tensor)
        reference_embedding = model_token_to_embedding(reference_indices)

        shap_n_samples = min(1000, 2**sample_tensor.shape[1])

        full_coeff = calculate_second_order_faithshap_interaction(model,
                                                                  sample_tensor,
                                                                  reference_indices,
                                                                  sample_tensor.shape[1],
                                                                  shap_n_samples,
                                                                  )
        p_bar_info = ""
        for subspace_limit in [0, 1, 2,4, 8, 16, 32]:
            args.subspace_limit = subspace_limit

            # calculate consistency
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
                np.matmul(second_order_masks, full_coeff) + model(reference_indices).detach().cpu().numpy() ).reshape(-1)

            p_bar_info = p_bar_info + f"{subspace_limit} {np.mean(np.abs(faithshap_result - model_truthful_values))} "

            eval(f"final_faithshap_output_{args.subspace_limit}").append(faithshap_result)
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

                # f and g
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
                        np.matmul(second_order_masks, full_coeff) + model(
                    reference_indices).detach().cpu().numpy()).reshape(-1)

                answer = np.mean((model_truthful_values - faithshap_result) * sigmas)

                eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}").append(faithshap_result)
                eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}").append(model_truthful_values)
                eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}").append(sigmas)
                eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}").append(answer)


    os.makedirs("faithshap", exist_ok=True)
    for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
        if args.split_end - args.split_start == 0 and args.split_start == 0:
            np.save(f"faithshap/faithshap_final_lasso_output_subspace{subspace_limit}_seed{args.seed}",
                    eval(f"final_faithshap_output_{subspace_limit}"))
            np.save(f"faithshap/faithshap_final_model_output_subspace{subspace_limit}_seed{args.seed}",
                    eval(f"final_model_output_{subspace_limit}"))
        else:
            np.save(
                f"faithshap/faithshap_final_lasso_output_subspace{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                eval(f"final_faithshap_output_{subspace_limit}"))
            np.save(
                f"faithshap/faithshap_final_model_output_subspace{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                eval(f"final_model_output_{subspace_limit}"))

    for C in C_range:
        for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
            if args.split_end - args.split_start == 0 and args.split_start == 0:
                np.save(
                    f"faithshap/faithshap_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"faithshap/faithshap_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"faithshap/faithshap_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"faithshap/faithshap_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}"))
            else:
                np.save(
                    f"faithshap/faithshap_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"faithshap/faithshap_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"faithshap/faithshap_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}"))
                np.save(
                    f"faithshap/faithshap_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}"))



