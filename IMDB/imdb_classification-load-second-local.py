import os
import torch
import random
import argparse
import spacy
import itertools

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchtext.legacy import data
from torchtext.legacy import datasets
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn import linear_model
from imdb_cnn_model import CNN, CNN_truncate
from itertools import product
from utils import train, evaluate, count_parameters, epoch_time, expand_basis_fun, paragraph_to_sentence
from scipy.special import comb

parser = argparse.ArgumentParser(description='consistent args')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--long_sentence_trucate', type=int, default=0, help='trucate size')
parser.add_argument('--modelpath', type=str, default="tut4-model-seed123-trucate-0.pt", help='model path')
parser.add_argument('--subspace_limit', type=int, default=0, help='subspace_limit')
parser.add_argument('--degree', type=int, default=2, help='degree')
parser.add_argument('--samples_min', type=int, default=2000, help='degree')

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

TEXT = data.Field(tokenize='spacy',
                  tokenizer_language='en_core_web_sm',
                  batch_first=True)
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

train_data, valid_data = train_data.split(random_state=random.seed(SEED))

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data,
                 max_size=MAX_VOCAB_SIZE,
                 vectors="glove.6B.100d",
                 unk_init=torch.Tensor.normal_)

LABEL.build_vocab(train_data)

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

if args.long_sentence_trucate == 0:
    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
else:
    model = CNN_truncate(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX,
                         args.long_sentence_trucate)

print(f'The model has {count_parameters(model):,} trainable parameters')

criterion = nn.BCELoss()

model = model.to(device)
criterion = criterion.to(device)

model.load_state_dict(torch.load(args.modelpath))

test_loss, test_acc = evaluate(args, model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

nlp = spacy.load('en_core_web_sm')


def predict_sentiment(model, sentence, min_len=5):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = model(tensor)
    return prediction.item()


def generate_random_mask(args, x, list_of_sentences, n_samples=1000, subspace_limit=0):
    # should return 1 and -1
    assert x.shape[0] == 1  # default: batch size
    length = x.shape[1]
    sentence_num = len(list_of_sentences)
    if subspace_limit > sentence_num:
        subspace_limit = sentence_num
    assert subspace_limit <= sentence_num  # maximum number of indexes of 0s

    if subspace_limit == 0:
        if n_samples == args.samples_min:
            mask_matrix = ((np.random.rand(n_samples, sentence_num) > .5) * 2 - 1).astype(int)
        else:
            mask_matrix = np.array(list(product([-1, 1], repeat=sentence_num)))
    else:  # subspace_limit is not 0
        if n_samples == args.samples_min:
            combnition_number_list = []
            for i in range(subspace_limit, 0, -1):
                comb_num = comb(sentence_num, i)
                if len(combnition_number_list) == 0 or comb_num / combnition_number_list[0] > 1 / n_samples:
                    combnition_number_list.append(comb_num)
            combnition_number_prob = combnition_number_list / sum(combnition_number_list)
            num_of_zeros = np.random.choice(np.arange(subspace_limit, subspace_limit - len(combnition_number_list), -1),
                                            n_samples, p=combnition_number_prob)
            column_index_every_row = [np.random.choice(sentence_num, num_of_zero, replace=False) for num_of_zero in
                                      num_of_zeros]

            mask_matrix = np.ones((n_samples, sentence_num))
            for _i in range(n_samples):
                mask_matrix[_i, column_index_every_row[_i]] = 0
            mask_matrix = mask_matrix * 2 - 1
        else:
            mask_matrix = np.array(list(product([0, 1], repeat=sentence_num)))
            mask_matrix = mask_matrix[np.where(mask_matrix.sum(axis=1) >= sentence_num - subspace_limit)[0],
                          :].squeeze()
            mask_matrix = mask_matrix * 2 - 1

    return mask_matrix


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

def text_to_str_sentence(text):
    return ' '.join(text)

def mask_to_masked_sample(masks_tensor, sample_tensor, sentence_index_to_word_index, pad_idx=1):
    sentence_length = sample_tensor.shape[1]
    return_tensor = []
    word_num_in_every_sentence = [len(value) for key, value in sentence_index_to_word_index.items()]
    assert len(sentence_index_to_word_index) == masks_tensor.shape[
        1], f"{len(sentence_index_to_word_index)}, {masks_tensor.shape}"
    for each_mask in masks_tensor:
        word_mask = [each_mask[_i].repeat(word_num_in_every_sentence[_i]) for _i in
                     range(len(sentence_index_to_word_index))]
        word_mask = torch.cat(word_mask)
        _tmp = torch.masked_select(sample_tensor, word_mask)
        _tmp = F.pad(_tmp, (0, sentence_length - len(_tmp)), "constant", pad_idx)
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

locality = 4

for test_index in pbar:
    one_test_data = test_data.__getitem__(test_index)
    if args.long_sentence_trucate != 0:
        one_test_sample = one_test_data.text[0:args.long_sentence_trucate]  # list of string words
    else:
        one_test_sample = one_test_data.text

    sentence_index_to_word_index, list_of_sentences = paragraph_to_sentence(one_test_sample)

    one_test_sample_length = len(list_of_sentences)
    n_samples = min(args.samples_min, 2 ** one_test_sample_length)
    one_test_sample_label = one_test_data.label

    basis = generate_random_mask(args,
                                 text_list_to_token_tensor(one_test_sample, length=len(one_test_sample)),
                                 list_of_sentences,
                                 n_samples=n_samples,
                                 subspace_limit=locality)  # 1s and -1s
    if len(list_of_sentences) < args.degree:
        continue

    sample_tensor = sentence_to_token_tensor(one_test_sample, length=len(one_test_sample))

    # build dataset
    masks_tensor = torch.from_numpy((basis + 1) / 2).cuda().bool()
    masked_samples_tensor = mask_to_masked_sample(masks_tensor, sample_tensor, sentence_index_to_word_index,
                                                  pad_idx=PAD_IDX)
    masked_samples_tensor = masked_samples_tensor.long()

    masked_samples_dataset = TensorDataset(masked_samples_tensor)
    masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=128, shuffle=False)

    values = []
    for _data in masked_samples_data_loader:
        values.append(model(_data[0]).detach().cpu())
    values = torch.cat(values).squeeze().numpy()  # (17000, 7)

    basis = np.array(basis)

    expanded_basis = expand_basis_fun(basis, args.degree)

    LassoSolver = linear_model.Lasso(fit_intercept=True, alpha=0.01)
    LassoSolver.fit(expanded_basis, values)
    coef = LassoSolver.coef_

    p_bar_info = ""
    for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
        args.subspace_limit = subspace_limit

        truthful_sample_basis = generate_random_mask(args,
                                                     text_list_to_token_tensor(one_test_sample,
                                                                               length=len(one_test_sample)),
                                                     list_of_sentences,
                                                     n_samples=n_samples,
                                                     subspace_limit=args.subspace_limit)  # 1s and -1s
        truthful_sample_masks = torch.from_numpy((truthful_sample_basis + 1) / 2).cuda().bool()

        # process model f output
        masked_samples_tensor = mask_to_masked_sample(truthful_sample_masks, sample_tensor,
                                                      sentence_index_to_word_index, pad_idx=PAD_IDX)
        masked_samples_tensor = masked_samples_tensor.long()

        masked_samples_dataset = TensorDataset(masked_samples_tensor)
        masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=128, shuffle=False)

        truthful_values = []
        for _data in masked_samples_data_loader:
            truthful_values.append(model(_data[0]).detach().cpu())
        model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

        expanded_truthful_sample_basis = expand_basis_fun(truthful_sample_basis, args.degree)

        scikit_lasso_result = (
                    np.matmul(expanded_truthful_sample_basis, coef.reshape(-1, 1)) + LassoSolver.intercept_).reshape(-1)

        pbar.set_description("sentence length: %d, truthful value: %.4f" % (
        one_test_sample_length, np.mean(np.abs(scikit_lasso_result - model_truthful_values))))

        eval(f"final_lasso_output_{subspace_limit}").append(scikit_lasso_result)
        eval(f"final_model_output_{subspace_limit}").append(model_truthful_values)

    pbar.set_description("sentence length: %d" % (one_test_sample_length))

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

        combination_mask = 1 - combination_mask  # number of 1s <= 3, this is kai_s

        # loop for subspace
        for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
            args.subspace_limit = subspace_limit
            truthful_sample_basis = generate_random_mask(args,
                                                         text_list_to_token_tensor(one_test_sample,
                                                                                   length=len(one_test_sample)),
                                                         list_of_sentences,
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
                                                          sentence_index_to_word_index, pad_idx=PAD_IDX)
            masked_samples_tensor = masked_samples_tensor.long()

            masked_samples_dataset = TensorDataset(masked_samples_tensor)
            masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=128, shuffle=False)

            truthful_values = []
            for _data in masked_samples_data_loader:
                truthful_values.append(model(_data[0]).detach().cpu())
            model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

            expanded_truthful_sample_basis = expand_basis_fun(truthful_sample_basis, args.degree)

            scikit_lasso_result = (
                    np.matmul(expanded_truthful_sample_basis,
                              coef.reshape(-1, 1)) + LassoSolver.intercept_).reshape(-1)

            answer = np.mean((model_truthful_values - scikit_lasso_result) * sigmas)

            eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}").append(scikit_lasso_result)
            eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}").append(model_truthful_values)
            eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}").append(sigmas)
            eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}").append(answer)

os.makedirs("harmonicalocal2degree", exist_ok=True)
for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
    if args.split_end - args.split_start == 0 and args.split_start == 0:
        np.save(f"harmonicalocal2degree/harmonicalocal2degree_final_lasso_output_subspace{subspace_limit}_seed{args.seed}",
                eval(f"final_lasso_output_{subspace_limit}"))
        np.save(f"harmonicalocal2degree/harmonicalocal2degree_final_model_output_subspace{subspace_limit}_seed{args.seed}",
                eval(f"final_model_output_{subspace_limit}"))
    else:
        np.save(
            f"harmonicalocal2degree/harmonicalocal2degree_final_lasso_output_subspace{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
            eval(f"final_lasso_output_{subspace_limit}"))
        np.save(
            f"harmonicalocal2degree/harmonicalocal2degree_final_model_output_subspace{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
            eval(f"final_model_output_{subspace_limit}"))

for C in C_range:
    for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
        if args.split_end - args.split_start == 0 and args.split_start == 0:
            np.save(f"harmonicalocal2degree/harmonicalocal2degree_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}"))
            np.save(f"harmonicalocal2degree/harmonicalocal2degree_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}"))
            np.save(f"harmonicalocal2degree/harmonicalocal2degree_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}"))
            np.save(f"harmonicalocal2degree/harmonicalocal2degree_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}",
                    eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}"))
        else:
            np.save(f"harmonicalocal2degree/harmonicalocal2degree_truthful_lasso_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                eval(f"final_truthful_lasso_C_{C}_subspace_{subspace_limit}"))
            np.save(f"harmonicalocal2degree/harmonicalocal2degree_truthful_model_output_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                eval(f"final_truthful_model_C_{C}_subspace_{subspace_limit}"))
            np.save(f"harmonicalocal2degree/harmonicalocal2degree_truthful_sigma_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                    eval(f"final_truthful_sigma_C_{C}_subspace_{subspace_limit}"))
            np.save(f"harmonicalocal2degree/harmonicalocal2degree_truthful_answer_C_{C}_subspace_{subspace_limit}_seed{args.seed}_{args.split_start}_{args.split_end}",
                eval(f"final_truthful_answer_C_{C}_subspace_{subspace_limit}"))
