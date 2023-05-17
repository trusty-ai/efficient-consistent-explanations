import os
import torch
import spacy
import random
import itertools
import argparse

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchtext.legacy import data
from torchtext.legacy import datasets
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from itertools import product
from imdb_cnn_model import CNN, CNN_truncate
from utils import train, evaluate, count_parameters, epoch_time, expand_basis_fun, paragraph_to_sentence
from scipy.special import comb
from captum.attr import LimeBase

parser = argparse.ArgumentParser(description='consistent args')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--long_sentence_trucate', type=int, default=0, help='trucate size')
parser.add_argument('--modelpath', type=str, default="tut4-model.pt", help='model path')
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

TEXT = data.Field(tokenize='spacy',
                  tokenizer_language='en_core_web_sm',
                  batch_first=True)
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

train_data, valid_data = train_data.split(random_state = random.seed(SEED))

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data,
                 max_size=MAX_VOCAB_SIZE,
                 vectors="glove.6B.100d",
                 unk_init=torch.Tensor.normal_)

LABEL.build_vocab(train_data)

BATCH_SIZE = 128

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

if args.split_end - args.split_start == 0 and args.split_start == 0:
    pbar = tqdm(range((len(test_data))))
else:
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
        # print('perturbed_inp', perturbed_inp)
        # print('norm', torch.norm(original_emb), torch.norm(perturbed_emb))
        distance = 1 - F.cosine_similarity(original_emb, perturbed_emb, dim=1)
        # print('distance', distance)

        return torch.exp(-1 * (distance ** 2) / 2)


    # binary vector where each word is selected independently and uniformly at random
    def bernoulli_perturb(text, **kwargs):
        binary_vec_sum = 0
        while binary_vec_sum == 0:
            probs = torch.ones((1, len(kwargs["list_of_sentences"]))) * 0.5
            binary_vec = torch.bernoulli(probs).long()
            binary_vec_sum = torch.sum(binary_vec)
        return binary_vec


    def interp_to_input(interp_sample, original_input, **kwargs):
        sentence_number = interp_sample.shape[1]
        sentence_index_to_word_index = kwargs["sentence_index_to_word_index"]
        word_num_in_every_sentence = [len(value) for key, value in sentence_index_to_word_index.items()]

        word_mask = [interp_sample[0][_i].repeat(word_num_in_every_sentence[_i]) for _i in
                     range(len(sentence_index_to_word_index))]
        word_mask = torch.cat(word_mask).reshape(1, -1)

        return original_input[word_mask.bool()].view(original_input.size(0), -1)


    def text_list_to_token_tensor(tokenized, length):
        if len(tokenized) < length:
            tokenized += ['<pad>'] * (length - len(tokenized))
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(0)
        return tensor


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
                num_of_zeros = np.random.choice(
                    np.arange(subspace_limit, subspace_limit - len(combnition_number_list), -1), n_samples,
                    p=combnition_number_prob)
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
                mask_matrix = mask_matrix.reshape(-1, sentence_num)
                mask_matrix = mask_matrix * 2 - 1
        return mask_matrix


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


    from captum._utils.models.linear_model import SkLearnLasso

    lasso_lime_base = LimeBase(
        forward_func,
        interpretable_model=SkLearnLasso(alpha=0.001),
        similarity_func=exp_embedding_cosine_distance,
        perturb_func=bernoulli_perturb,
        perturb_interpretable_space=True,
        from_interp_rep_transform=interp_to_input,
        to_interp_rep_transform=None,
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

        sentence_index_to_word_index, list_of_sentences = paragraph_to_sentence(one_test_sample)

        sample_tensor = text_list_to_token_tensor(one_test_sample, length=len(one_test_sample))
        lime_coeff = lasso_lime_base.attribute(
            sample_tensor,  # add batch dimension for Captum
            n_samples=min(args.samples_min, 2 ** len(list_of_sentences)),
            show_progress=False,
            sentence_index_to_word_index=sentence_index_to_word_index,
            list_of_sentences=list_of_sentences,
        )  # same shape with sample_tensor

        p_bar_info = ""
        for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
            args.subspace_limit = subspace_limit

            n_samples = min(args.samples_min, 2 ** len(list_of_sentences))
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

            lime_result = (
                    np.matmul(truthful_sample_masks.cpu().numpy(), lime_coeff.reshape(-1,
                                                                                      1).cpu().numpy()) + lasso_lime_base.interpretable_model.bias().cpu().numpy()).reshape(
                -1)

            p_bar_info = p_bar_info + f"{subspace_limit} {np.mean(np.abs(lime_result - model_truthful_values))} "

            eval(f"final_lime_output_{subspace_limit}").append(lime_result)
            eval(f"final_model_output_{subspace_limit}").append(model_truthful_values)
        pbar.set_description("sentence length: %d" % (len(list_of_sentences)))

        # loop for C
        for C in C_range:
            x = np.arange(len(list_of_sentences))
            combination_mask_index_under_degree = []
            truthful_degree = np.arange(0, C + 1).astype(int)
            for each_limit in truthful_degree:
                combination_mask_index_under_degree += list(itertools.combinations(x, each_limit))
            combination_mask = np.ones((len(combination_mask_index_under_degree), len(list_of_sentences)))
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
                num_of_zeros = np.random.choice(
                    np.arange(subspace_limit, subspace_limit - len(combnition_number_list), -1), n_samples,
                    p=combnition_number_prob)
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
                mask_matrix = mask_matrix.reshape(-1, sentence_num)
                mask_matrix = mask_matrix * 2 - 1

        return mask_matrix


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

    def recover_full_size_mask(truthful_sample_masks, sentence_index_to_word_index):
        word_num_in_every_sentence = [len(value) for key, value in sentence_index_to_word_index.items()]
        return_tensor = []
        for i in range(len(word_num_in_every_sentence)):
            return_tensor.append(truthful_sample_masks[:, i:i+1].repeat(1, word_num_in_every_sentence[i]))
        return_tensor = torch.hstack(return_tensor)

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

        sentence_index_to_word_index, list_of_sentences = paragraph_to_sentence(one_test_sample)

        sample_tensor = text_list_to_token_tensor(one_test_sample,
                                                  length=max(len(one_test_sample), 5))  # greater than kernel size
        reference_indices = token_reference.generate_reference(sample_tensor.shape[1], device=device).unsqueeze(0)


        attributions_ig, delta = interpret_sentence(model, sample_tensor, reference_indices)
        attributions_ig = attributions_ig.sum(dim=2)[:, 0: sample_tensor.shape[1]]

        p_bar_info = ""
        for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:
            args.subspace_limit = subspace_limit

            # calculate consistency
            n_samples = min(args.samples_min, 2 ** len(list_of_sentences))
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

            # recover full size mask
            truthful_sample_masks_full_size = recover_full_size_mask(truthful_sample_masks, sentence_index_to_word_index)

            masked_samples_dataset = TensorDataset(masked_samples_tensor)
            masked_samples_data_loader = DataLoader(masked_samples_dataset, batch_size=128, shuffle=False)

            truthful_values = []
            for _data in masked_samples_data_loader:
                truthful_values.append(model(_data[0]).detach().cpu())
            model_truthful_values = torch.cat(truthful_values).squeeze().numpy()

            ig_result = (
                    np.matmul(truthful_sample_masks_full_size.cpu().numpy(),
                              attributions_ig.reshape(-1, 1).cpu().numpy()) + model(
                reference_indices).detach().cpu().numpy() - delta.cpu().numpy()).reshape(-1)
            p_bar_info = p_bar_info + f"{subspace_limit} {np.mean(np.abs(ig_result - model_truthful_values))} "


            eval(f"final_ig_output_{args.subspace_limit}").append(ig_result)
            eval(f"final_model_output_{args.subspace_limit}").append(model_truthful_values)
        pbar.set_description("sentence length: %d" % len(list_of_sentences))

        # loop for C
        for C in C_range:
            x = np.arange(len(list_of_sentences))
            combination_mask_index_under_degree = []
            truthful_degree = np.arange(0, C + 1).astype(int)
            for each_limit in truthful_degree:
                combination_mask_index_under_degree += list(itertools.combinations(x, each_limit))
            combination_mask = np.ones((len(combination_mask_index_under_degree), len(list_of_sentences)))
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

                truthful_sample_masks_full_size = recover_full_size_mask(truthful_sample_masks,
                                                                         sentence_index_to_word_index)
                ig_result = (
                        np.matmul(truthful_sample_masks_full_size.cpu().numpy(),
                                  attributions_ig.reshape(-1, 1).cpu().numpy()) + model(
                    reference_indices).detach().cpu().numpy() - delta.cpu().numpy()).reshape(-1)

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
                num_of_zeros = np.random.choice(
                    np.arange(subspace_limit, subspace_limit - len(combnition_number_list), -1), n_samples,
                    p=combnition_number_prob)
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

        sentence_index_to_word_index, list_of_sentences = paragraph_to_sentence(one_test_sample)

        if len(list_of_sentences) == 1:
            continue  # kernel shap does not support

        sample_tensor = text_list_to_token_tensor(one_test_sample,
                                                  length=max(len(one_test_sample), 5))  # greater than kernel size
        reference_indices = token_reference.generate_reference(sample_tensor.shape[1], device=device).unsqueeze(0)

        word_num_in_every_sentence = [len(value) for key, value in sentence_index_to_word_index.items()]
        feature_mask = torch.cat([torch.Tensor([i]).repeat(word_num_in_every_sentence[i]) for i in
                                  range(len(word_num_in_every_sentence))]).unsqueeze(0)

        attr = ks.attribute(inputs=sample_tensor,
                            baselines=reference_indices,
                            feature_mask=feature_mask.long().to(device),
                            target=None,
                            n_samples=min(args.samples_min, 2 ** len(list_of_sentences)),
                            return_input_shape=False)

        p_bar_info = ""
        for subspace_limit in [0, 1, 2, 4, 8, 16, 32]:

            args.subspace_limit = subspace_limit

            # calculate consistency
            n_samples = min(args.samples_min, 2 ** len(list_of_sentences))
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

            ks_result = (
                    np.matmul(truthful_sample_masks.cpu().numpy(),
                              attr.reshape(-1, 1).cpu().numpy()) + model(reference_indices).detach().cpu().numpy()  ).reshape(-1)

            p_bar_info = p_bar_info + f"{subspace_limit} {np.mean(np.abs(ks_result - model_truthful_values))} "

            pbar.set_description("sentence length: %d, truthful value: %.4f" % (
                sample_tensor.shape[1], np.mean(np.abs(ks_result - model_truthful_values))))

            eval(f"final_ks_output_{args.subspace_limit}").append(ks_result)
            eval(f"final_model_output_{args.subspace_limit}").append(model_truthful_values)
        pbar.set_description("sentence length: %d" % (len(list_of_sentences)))

        # loop for C
        for C in C_range:
            x = np.arange(len(list_of_sentences))
            combination_mask_index_under_degree = []
            truthful_degree = np.arange(0, C + 1).astype(int)
            for each_limit in truthful_degree:
                combination_mask_index_under_degree += list(itertools.combinations(x, each_limit))
            combination_mask = np.ones((len(combination_mask_index_under_degree), len(list_of_sentences)))
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
