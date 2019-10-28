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
        super(LSTM_NER_TeacherForcing, self).__init__()

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
            nn.LogSoftmax(dim=1))

    def forward(self, x, teacher_forcing=False, truth=None):
        # Embedding
        emb = self.embedding(x)

        # RNN Encoder
        output, _ = self.rnn(emb)

        print(output.shape)

        # Sequence length
        batch_size = output.shape[0]
        seq_len = output.shape[1]

        # Force past predicted to classifier
        y = torch.zeros(batch_size, 0, self.classes)
        print(seq_len)

        past = torch.zeros(batch_size, 1)

        for time in range(seq_len):

            t_slice = output[:,time, :].squeeze(1)

            print(past.shape)
            print(t_slice.shape)

            if teacher_forcing and time > 0:
                past = truth[:, time - 1]
            
            # Concat with past
            t_concat = torch.cat((past, t_slice), dim=1).unsqueeze(1)
            
            print(t_concat.shape)

            # Classifier
            y_t = self.classifier(t_concat)

            print(y_t.shape)
            print(y.shape)

            y = torch.cat((y, y_t), dim=1)

            print(y.shape)
            input()


        # Reshape for loss
        y = torch.transpose(y, 1, 2)

        print(y.shape)
        input()

        return y
