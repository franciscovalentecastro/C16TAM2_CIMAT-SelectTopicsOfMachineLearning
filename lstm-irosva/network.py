# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class LSTM_MTL_Classifier(nn.Module):

    def __init__(self, embedding_size, hidden_size,
                 vocab_size, layers, dropout, categories):
        super(LSTM_MTL_Classifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.irony = LSTM_Irony_Classifier(embedding_size,
                                           hidden_size,
                                           layers,
                                           dropout)
        self.topic = LSTM_Topic_Classifier(embedding_size,
                                           hidden_size,
                                           layers,
                                           dropout,
                                           categories)

    def forward(self, x):
        # Embedding
        x = torch.transpose(x, 0, 1)
        emb = self.embedding(x)

        return (self.irony(emb), self.topic(emb))


class LSTM_Irony_Classifier(nn.Module):

    def __init__(self, embedding_size, hidden_size,
                 layers, dropout):
        super(LSTM_Irony_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers

        self.lstm = nn.LSTM(embedding_size, hidden_size, layers,
                            dropout=dropout, batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(hidden_size * layers * 2, 1)

    def forward(self, x):
        # RNN
        output, (hidden_n, cell_n) = self.lstm(x)
        hidden_n = torch.transpose(hidden_n, 0, 1)
        hidden_n = hidden_n.reshape(-1, self.hidden_size * self.layers * 2)

        # Binary classifier
        y = self.linear(hidden_n)
        y = torch.sigmoid(y)
        return y.view(-1)


class LSTM_Topic_Classifier(nn.Module):

    def __init__(self, embedding_size, hidden_size,
                 layers, dropout, categories):
        super(LSTM_Topic_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers

        self.lstm = nn.LSTM(embedding_size, hidden_size, layers,
                            dropout=dropout, batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(hidden_size * layers * 2, categories)
        self.logsoftmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        # RNN
        output, (hidden_n, cell_n) = self.lstm(x)
        hidden_n = torch.transpose(hidden_n, 0, 1)
        hidden_n = hidden_n.reshape(-1, self.hidden_size * self.layers * 2)

        # Classifier
        y = self.linear(hidden_n)
        y = self.logsoftmax(y)
        return y
