import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def train(args, model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        text, label = batch.text, batch.label
        if args.long_sentence_trucate != 0:
            text = text[:, :args.long_sentence_trucate]
        #         label = substitue_label(label)
        optimizer.zero_grad()
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(args, model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:
            text, label = batch.text, batch.label
            if args.long_sentence_trucate != 0:
                text = text[:, :args.long_sentence_trucate]
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, label)
            acc = binary_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

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

TERMINATION_SIGN = [".", "!", "?"]

def paragraph_to_sentence(list_of_words):
    sentence_index_to_word_index = {}
    head = 0
    sentence_head = 0
    for i in range(len(list_of_words)):
        each_word = list_of_words[i]
        if each_word in TERMINATION_SIGN or i==len(list_of_words)-1:
            sentence_index_to_word_index[sentence_head] = np.arange(head, i + 1, 1)
            sentence_head += 1
            head = i + 1

    list_of_sentences = []
    for key, value in sentence_index_to_word_index.items():
        list_of_sentences.append(np.array(list_of_words)[value].tolist())

    return sentence_index_to_word_index, list_of_sentences
