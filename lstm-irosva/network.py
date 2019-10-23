# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class LSTM_MTL_Classifier(nn.Module):

    def __init__(self, embedding_size, hidden_size,
                 vocab_size, layers, dropout, categories):
        super(LSTM_MTL_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # Encoder
        self.rnn = nn.LSTM(embedding_size, hidden_size, layers,
                           dropout=dropout, batch_first=True,
                           bidirectional=True)

        # Classifiers
        self.irony = nn.Sequential(
            nn.Linear(hidden_size * layers * 2, 1),
            nn.Sigmoid())

        self.topic = nn.Sequential(
            nn.Linear(hidden_size * layers * 2, categories),
            nn.LogSoftmax(dim=0))

        self.humor = nn.Sequential(
            nn.Linear(hidden_size * layers * 2, 1),
            nn.Sigmoid())

    def forward(self, x):
        # Embedding
        x = torch.transpose(x, 0, 1)
        emb = self.embedding(x)

        # RNN
        output, (hidden_n, cell_n) = self.rnn(emb)
        hidden_n = torch.transpose(hidden_n, 0, 1)
        hidden_n = hidden_n.reshape(-1, self.hidden_size * self.layers * 2)

        return (self.irony(hidden_n).view(-1),
                self.topic(hidden_n),
                self.humor(hidden_n).view(-1))


class LSTM_Irony_Classifier(nn.Module):

    def __init__(self, embedding_size, hidden_size,
                 vocab_size, layers, dropout):
        super(LSTM_Irony_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # Encoder
        self.rnn = nn.LSTM(embedding_size, hidden_size, layers,
                           dropout=dropout, batch_first=True,
                           bidirectional=True)

        # Classifier
        self.irony = nn.Sequential(
            nn.Linear(hidden_size * layers * 2, 1),
            nn.Sigmoid())

    def forward(self, x):
        # Embedding
        x = torch.transpose(x, 0, 1)
        emb = self.embedding(x)

        # RNN
        output, (hidden_n, cell_n) = self.rnn(emb)
        hidden_n = torch.transpose(hidden_n, 0, 1)
        hidden_n = hidden_n.reshape(-1, self.hidden_size * self.layers * 2)

        return self.irony(hidden_n).view(-1)
