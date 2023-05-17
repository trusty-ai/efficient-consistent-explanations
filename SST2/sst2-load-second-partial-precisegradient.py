import os
import math
import spacy
import torch
import random
import argparse

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from itertools import product
from torchtext.legacy import data
from sst2_cnn_model import CNN, CNN_truncate
from collections import Counter
from utils import train, evaluate, count_parameters, epoch_time, expand_basis_fun, paragraph_to_sentence
from scipy.special import comb
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='consistent args')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--long_sentence_trucate', type=int, default=50, help='trucate size')
parser.add_argument('--modelpath', type=str, default="sst2-model-epoch4.pt", help='model path')
parser.add_argument('--subspace_limit', type=int, default=0, help='subspace_limit')
parser.add_argument('--degree', type=int, default=2, help='degree')
parser.add_argument('--samples_min', type=int, default=2000, help='degree')
parser.add_argument('--n', type=int, default=3, help='n anchors')
parser.add_argument('--ep_consistent_loss', type=float, default=1, help='ep_consistent_loss')
parser.add_argument('--fit_epochs', type=int, default=1000, help='fit_epochs')

parser.add_argument('--split_start', type=int, default=0, help='accelerate', required=False)
parser.add_argument('--split_end', type=int, default=0, help='accelerate', required=False)

args = parser.parse_args()

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

def generate_random_mask(args, x, n_samples=1000, subspace_limit=0, anchor=None):
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
    else:  # subspace_limit is not 0, return a matrix with number of 0 in each line less than args.subspace_limit
        if n_samples == args.samples_min:
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

    if anchor is not None:  # ensure ther are at least 1 basis assigned to each anchor
        mask_matrix = np.vstack([anchor, mask_matrix])

    return np.float32(mask_matrix)

def generate_local_mask(args, x, n_samples=1000, subspace_limit=0):
    # should return 1 and -1
    assert x.shape[0] == 1  # default: batch size
    length = x.shape[1]
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

def generate_random_anchor(n_minus_1, x):
    assert x.shape[0] == 1  # default: batch size
    length = x.shape[1]
    anchor_matrix = ((np.random.rand(n_minus_1, length) > .5) * 2 - 1).astype(int)
    anchor_matrix = np.vstack([np.ones([1, length]), anchor_matrix])

    return anchor_matrix

def assign_basis_to_anchor(basis, anchor_matrix):
    anchor_index_list = []
    # compute distance
    for each_basis in basis:
        each_distance = np.sum(np.abs(anchor_matrix - each_basis), axis=1)
        each_index = np.min( np.where(each_distance == each_distance.min())[0] )
        anchor_index_list.append(each_index)
    return np.array(anchor_index_list)

def group_expanded_basis(n, expanded_basis, anchor_index_list, values):
    grouped_expanded_basis_dict = {}
    grouped_value_dict = {}
    for each_index, each_basis, each_value in zip(anchor_index_list, expanded_basis, values):
        if each_index in grouped_expanded_basis_dict:
            grouped_expanded_basis_dict[each_index].append(each_basis)
            grouped_value_dict[each_index].append(each_value)
        else:
            grouped_expanded_basis_dict[each_index] = [each_basis]
            grouped_value_dict[each_index] = [each_value]

    for each in grouped_expanded_basis_dict:
        grouped_expanded_basis_dict[each] = np.float32(np.array(grouped_expanded_basis_dict[each]))
    for each in grouped_value_dict:
        grouped_value_dict[each] = np.float32(np.array(grouped_value_dict[each]))

    return grouped_expanded_basis_dict, grouped_value_dict


def text_list_to_token_tensor(tokenized, length):
    if len(tokenized) < length:
        tokenized += ['<pad>'] * (length - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    return tensor

def sentence_to_token_tensor(tokenized, length):
    if len(tokenized) < length:
        tokenized += ['<pad>'] * (length - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    return tensor

def numpy_to_device(numpy_array, device):
    return torch.from_numpy(numpy_array).to(device)

def text_to_str_sentence(text):
    return ' '.join(text)

def mask_to_masked_sample(masks_tensor, sample_tensor, pad_idx=1):
    sentence_length = sample_tensor.shape[1]
    return_tensor = []
    for each_mask in masks_tensor:
        _tmp = torch.masked_select(sample_tensor, each_mask)
        _tmp = F.pad(_tmp, (0, sentence_length - torch.sum(each_mask)), "constant", pad_idx)
        return_tensor.append(_tmp)

    return_tensor = torch.vstack(return_tensor)
    return return_tensor


degree = args.degree

# loop in test loader
final_lasso_output_0 = []
final_model_output_0 = []
final_lasso_output_1 = []
final_model_output_1 = []
final_lasso_output_2 = []
final_model_output_2 = []
final_lasso_output_4 = []
final_model_output_4 = []
final_lasso_output_8 = []
final_model_output_8 = []
final_lasso_output_16 = []
final_model_output_16 = []
final_lasso_output_32 = []
final_model_output_32 = []

# init variables
C_range = [1, 2, 3]
for C in C_range:
    for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
        exec(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit} = []")
        exec(f"final_truthful_model_C_{C}_subspace_{subspace_limit} = []")
        exec(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit} = []")
        exec(f"final_truthful_answer_C_{C}_subspace_{subspace_limit} = []")

if args.split_end - args.split_start == 0 and args.split_start == 0:
    pbar = tqdm(range((len(test_data))))
else:
    pbar = tqdm(range(args.split_start, args.split_end))

for test_index in pbar:
    one_test_data = test_data.__getitem__(test_index)
    if args.long_sentence_trucate != 0:
        one_test_sample = one_test_data.text[0:args.long_sentence_trucate]  # list of string words
    else:
        one_test_sample = one_test_data.text

    one_test_sample_length = len(one_test_sample)
    n_samples = min(args.samples_min, 2 ** one_test_sample_length)
    one_test_sample_label = one_test_data.label

    if len(one_test_sample) < args.degree:
        continue

    anchor = generate_random_anchor(args.n-1, text_list_to_token_tensor(one_test_sample, length=len(one_test_sample)))

    local_basis_sample_number = min(comb(one_test_sample_length, 4), n_samples / args.n)
    local_basis = generate_local_mask(args,
                                      text_list_to_token_tensor(one_test_sample, length=len(one_test_sample)),
                                      n_samples=int(local_basis_sample_number),
                                      # subspace_limit=max( min(4, int(np.ceil(np.log(one_test_sample_length)))), 1 ),
                                      subspace_limit=4
                                      )

    basis = generate_random_mask(args,
                                 text_list_to_token_tensor(one_test_sample, length=len(one_test_sample)),
                                 n_samples=n_samples,
                                 subspace_limit=args.subspace_limit)  # 1s and -1s
    basis = np.vstack([local_basis, basis])

    anchor_index_list = assign_basis_to_anchor(basis, anchor)  # np.array
    print(anchor_index_list)
    print(Counter(anchor_index_list))

    sample_tensor = sentence_to_token_tensor(one_test_sample, length=len(one_test_sample))

    # build dataset
    masks_tensor = torch.from_numpy((basis + 1) / 2).cuda().bool()
    masked_samples_tensor = mask_to_masked_sample(masks_tensor, sample_tensor,
                                                  pad_idx=PAD_IDX)
    masked_samples_tensor = masked_samples_tensor.long()

    masked_samples_dataset = TensorDataset(masked_samples_tensor)
    masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=512, shuffle=False)

    values = []
    for _data in masked_samples_data_loader:
        values.append(model(_data[0]).detach().cpu())
    values = torch.cat(values).squeeze().numpy()  # (17000, 7)

    basis = np.array(basis)
    print("basis shape", basis.shape)

    expanded_basis = expand_basis_fun(basis, args.degree)
    # add intercept column to the 1st column
    expanded_basis = np.hstack([np.ones((expanded_basis.shape[0], 1)), expanded_basis])

    # group_expanded_basis
    grouped_expanded_basis_dict, grouped_value_dict = group_expanded_basis(args.n, expanded_basis, anchor_index_list, values)
    for each in grouped_expanded_basis_dict:
        grouped_expanded_basis_dict[each] = numpy_to_device(grouped_expanded_basis_dict[each], device)
    for each in grouped_value_dict:
        grouped_value_dict[each] = numpy_to_device(grouped_value_dict[each], device)




    anchor_params = Variable(torch.randn((args.n, expanded_basis.shape[1])).to(device), requires_grad=True)

    def cosine_learning_rate(epoch):
        eta_min = 0
        eta_max = 10
        return eta_min + (1/2) * (eta_max-eta_min) * (1 + np.cos(epoch / args.fit_epochs * math.pi))

    for epoch in range(args.fit_epochs):
        ep_total_loss = 0
        ep_fit_loss = 0
        ep_regularize_loss = 0
        ep_consistent_loss = 0
        for each_index in grouped_expanded_basis_dict:
            pred = torch.sum(grouped_expanded_basis_dict[each_index] * anchor_params[each_index], dim=1)
            ep_fit_loss += torch.sum(torch.abs(pred - grouped_value_dict[each_index]))
        ep_fit_loss = torch.sqrt(ep_fit_loss / expanded_basis.shape[0])
        ep_regularize_loss = torch.mean(torch.norm(anchor_params[:, 1:], dim=1, p=1))
        ep_consistent_loss = torch.norm(anchor_params[:, 1:] - torch.mean(anchor_params[:, 1:], dim=0, keepdim=True), dim=1).mean()

        if epoch % 100 == 0 or epoch==args.fit_epochs-1:
            print("%.5f %.2f %.2f" % (ep_fit_loss, ep_regularize_loss, ep_consistent_loss))
        ep_total_loss = ep_fit_loss + 0.01 * ep_regularize_loss + args.ep_consistent_loss * ep_consistent_loss

        grad = torch.autograd.grad(ep_total_loss, anchor_params, retain_graph=False)[0]
        grad = grad / torch.norm(grad)

        anchor_params = anchor_params - cosine_learning_rate(epoch) * grad

    anchor_params = anchor_params.detach().cpu().numpy()
    print("anchor_params", anchor_params)


    p_bar_info = ""
    for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:

        args.subspace_limit = subspace_limit

        truthful_sample_basis = generate_random_mask(args,
                                                     text_list_to_token_tensor(one_test_sample,
                                                                               length=len(one_test_sample)),
                                                     n_samples=n_samples,
                                                     subspace_limit=args.subspace_limit)  # 1s and -1s
        truthful_sample_anchor_index_list = assign_basis_to_anchor(truthful_sample_basis, anchor)  # np.array

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

        expanded_truthful_sample_basis = expand_basis_fun(truthful_sample_basis, args.degree)
        # add intercept column to the 1st column
        expanded_truthful_sample_basis = np.hstack([np.ones((expanded_truthful_sample_basis.shape[0], 1)), expanded_truthful_sample_basis])

        scikit_lasso_result = []
        for _i, each_index in enumerate(truthful_sample_anchor_index_list):
            scikit_lasso_result.append(np.sum(expanded_truthful_sample_basis[_i] * anchor_params[each_index]))
        scikit_lasso_result = np.array(scikit_lasso_result).reshape(-1)

        p_bar_info = p_bar_info + f"{subspace_limit} {np.mean(np.abs(scikit_lasso_result - model_truthful_values))} "

        eval(f"final_lasso_output_{subspace_limit}").append(scikit_lasso_result)
        eval(f"final_model_output_{subspace_limit}").append(model_truthful_values)

    pbar.set_description("sentence length: %d" % (sample_tensor.shape[1],))

os.makedirs(f"harmonica2degreepartial_precisegradient_sample{args.samples_min}_anchor{args.n}_consisloss{args.ep_consistent_loss}", exist_ok=True)
for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
    if args.split_end - args.split_start == 0 and args.split_start == 0:
        np.save(f"harmonica2degreepartial_precisegradient_sample{args.samples_min}_anchor{args.n}_consisloss{args.ep_consistent_loss}/harmonica2degreepartial_final_lasso_output_subspace{subspace_limit}_seed{args.seed}",
                eval(f"final_lasso_output_{subspace_limit}"))
        np.save(f"harmonica2degreepartial_precisegradient_sample{args.samples_min}_anchor{args.n}_consisloss{args.ep_consistent_loss}/harmonica2degreepartial_final_model_output_subspace{subspace_limit}_seed{args.seed}",
                eval(f"final_model_output_{subspace_limit}"))
    else:
        np.save(
            f"harmonica2degreepartial_precisegradient_sample{args.samples_min}_anchor{args.n}_consisloss{args.ep_consistent_loss}/harmonica2degreepartial_final_lasso_output_subspace{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
            eval(f"final_lasso_output_{subspace_limit}"))
        np.save(
            f"harmonica2degreepartial_precisegradient_sample{args.samples_min}_anchor{args.n}_consisloss{args.ep_consistent_loss}/harmonica2degreepartial_final_model_output_subspace{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
            eval(f"final_model_output_{subspace_limit}"))



