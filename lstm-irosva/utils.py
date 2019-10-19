# -*- coding: utf-8 -*-
from torchtext.data import *

# Import network
from network import *


def load_dataset(path, args):

    def tokenize(x):
        return x.split()

    TEXT = Field(sequential=True,
                 tokenize=tokenize,
                 lower=True)
    TOPIC = Field(sequential=False,
                  use_vocab=True,
                  lower=True)
    LABEL = Field(sequential=False,
                  use_vocab=False)

    datafields = [("id", None),
                  ("topic", TOPIC),
                  ("is_ironic", LABEL),
                  ("message", TEXT)]

    dataset = TabularDataset(path=path,
                             format='CSV',
                             skip_header=True,
                             fields=datafields)

    args.TEXT = TEXT
    args.TEXT.build_vocab(dataset)  # min_freq=2, max_size=args.vocabulary)

    args.TOPIC = TOPIC
    args.TOPIC.build_vocab(dataset)

    print('vocabulary length : ', len(args.TEXT.vocab))
    print('number of topics : ', len(args.TOPIC.vocab))

    return(dataset)


def print_batch(batch, targets, predicted, args):
    text = batch.__dict__['message']
    topic = batch.__dict__['topic']

    for idx, t in enumerate(torch.t(text)):
        # Topic
        topic_word = str(args.TOPIC.vocab.itos[int(topic[idx])])
        print('TOPIC : ({}){}'
              .format(int(topic[idx]), topic_word),
              end=', ')

        # Irony
        print('IS_IRONIC : ', int(targets[idx]), end=', ')
        print('PREDICTED : ', int(predicted[idx]))

        # Message
        print(str(idx) + ' MESSAGE : ', end='')
        for elem in t:
            word = str(args.TEXT.vocab.itos[int(elem)])
            if word != '<pad>':
                print('{}'.format(word), end=' ')
