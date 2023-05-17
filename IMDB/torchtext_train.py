import time
import torch
import random
import argparse

import numpy as np
import torch.nn as nn
import torch.optim as optim

from imdb_cnn_model import CNN, CNN_truncate
from torchtext.legacy import data
from torchtext.legacy import datasets
from utils import train, evaluate, count_parameters, epoch_time

parser = argparse.ArgumentParser(description='consistent args')
parser.add_argument('--seed', type=int, default=123, help='random seed', required=False)
parser.add_argument('--long_sentence_trucate', type=int, default=0, help='trucate size', required=False)

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

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

train_data, valid_data = train_data.split(random_state = random.seed(SEED))


test_examples = test_data.examples
all_test_labels = []
for i in range(len(test_examples)):
    all_test_labels.append(test_examples[i].label)
train_examples = train_data.examples
all_train_labels = []
for i in range(len(train_examples)):
    all_train_labels.append(train_examples[i].label)

pos_count = 0
neg_count = 0
for each in all_test_labels:
    if each=='pos':
        pos_count += 1
    else:
        neg_count += 1

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
    batch_size = BATCH_SIZE, 
    device = device)

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

pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

model = model.to(device)
criterion = criterion.to(device)


N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(args, model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(args, model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut4-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

model.load_state_dict(torch.load('tut4-model.pt'))
test_loss, test_acc = evaluate(args, model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
