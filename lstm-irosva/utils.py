# -*- coding: utf-8 -*-

from torchtext.data import *

# Import network
from network import *


def load_dataset(path, args):

    def tokenize(x):
        return x.split()

    TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
    LABEL = Field(sequential=False, use_vocab=False)

    datafields = [("id", None),
                  ("topic", None),
                  ("is_ironic", LABEL),
                  ("message", TEXT)]

    dataset = TabularDataset(path=path,
                             format='CSV',
                             skip_header=True,
                             fields=datafields)

    args.TEXT = TEXT
    args.TEXT.build_vocab(dataset, max_size=args.vocabulary)
    print('vocabulary length : ', len(args.TEXT.vocab))

    return(dataset)


def print_batch(batch, targets, predicted, args):
    text = batch.__dict__['message']

    for idx, t in enumerate(torch.t(text)):
        print(str(idx) + ' MESSAGE : ', end='')
        for elem in t:
            word = str(args.TEXT.vocab.itos[int(elem)])
            if word != '<pad>':
                print('{}'.format(word), end=' ')
        print('IS_IRONIC : ', int(targets[idx]), end=', ')
        print('PREDICTED : ', int(predicted[idx]))
