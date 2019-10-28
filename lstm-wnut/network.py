# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class LSTM_NER(nn.Module):

    def __init__(self, embedding_size, hidden_size,
                 vocab_size, layers, dropout, classes):
        super(LSTM_NER, self).__init__()

        # Parameters
        self.layers = layers
        self.classes = classes
        self.hidden_size = hidden_size

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # Encoder
        self.rnn = nn.LSTM(embedding_size, hidden_size, layers,
                           dropout=dropout, batch_first=True,
                           bidirectional=True)

        # Classifiers
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, classes),
            nn.LogSoftmax(dim=2))

    def forward(self, x):
        # Embedding
        emb = self.embedding(x)

        # RNN
        output, _ = self.rnn(emb)

        # Classifier
        y = self.classifier(output)

        # Reshape for loss
        y = torch.transpose(y, 1, 2)

        return y


class LSTM_NER_TeacherForcing(nn.Module):

    def __init__(self, embedding_size, hidden_size,
                 vocab_size, layers, dropout, classes):
        super(LSTM_NER, self).__init__()

        # Parameters
        self.layers = layers
        self.classes = classes
        self.hidden_size = hidden_size

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # Encoder
        self.rnn = nn.LSTM(embedding_size, hidden_size, layers,
                           dropout=dropout, batch_first=True,
                           bidirectional=True)

        # Classifier + previous tag
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size + 1, classes),
            nn.LogSoftmax(dim=2))

    def forward(self, x):
        # Embedding
        emb = self.embedding(x)

        # RNN
        output, _ = self.rnn(emb)

        # Classifier
        y = self.classifier(output)

        # Reshape for loss
        y = torch.transpose(y, 1, 2)

        return y
