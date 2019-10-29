# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class loss_ner():
    def __init__(self, args):
        # Network to train
        self.network = args.network

        # Weights for classifiying NER (BIO)
        w = torch.tensor([0, 0, 0, 0, .4, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1],
                         dtype=torch.float).to(args.device)

        # Set loss functions
        self.nllloss = torch.nn.NLLLoss(weight=w)
        self.bceloss = torch.nn.BCELoss()

    def loss(self, outputs, targets):
        if self.network == 'mtl' or \
           self.network == 'prop':

            ne, is_ne = outputs
            t_ne, t_is_ne = targets

            return self.nllloss(ne, t_ne) + \
                self.bceloss(is_ne, t_is_ne.float())
        else:
            return self.nllloss(outputs, targets)


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


class LSTM_NER_TF(nn.Module):

    def __init__(self, embedding_size, hidden_size, vocab_size,
                 layers, dropout, classes, device='cpu'):
        super(LSTM_NER_TF, self).__init__()

        # Parameters
        self.layers = layers
        self.classes = classes
        self.device = device
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

    def forward(self, x, teacher_forcing=False, truth=None):
        # Embedding
        emb = self.embedding(x)

        # RNN Encoder
        output, _ = self.rnn(emb)

        # batch size & sequence length
        batch_size = output.shape[0]
        seq_len = output.shape[1]

        if teacher_forcing:

            # past truth labels to one hot
            past = torch.zeros(batch_size, 1).long().to(self.device)
            past = torch.cat((past, truth[:, :-1]), dim=1).float().unsqueeze(2)
            concat = torch.cat((output, past), dim=2)

            # classifiy concat
            y = self.classifier(concat)

        else:

            # Force past predicted to classifier
            y = torch.zeros(batch_size, 0, self.classes).to(self.device)

            # Initial past prediction is zero
            past = torch.zeros(batch_size, 1, 1).to(self.device)

            # Classify using teacher forcing
            for time in range(seq_len):

                # get current time slice
                t_slice = output[:, time: time + 1, :]

                # Concat with past
                t_concat = torch.cat((t_slice, past), dim=2)

                # Classifier
                y_t = self.classifier(t_concat)
                y = torch.cat((y, y_t), dim=1)

                # preparing for next step
                past = (y_t.argmax(dim=2)).float().unsqueeze(2)

        # Reshape for loss
        y = torch.transpose(y, 1, 2)

        return y


class LSTM_NER_MTL(nn.Module):

    def __init__(self, embedding_size, hidden_size,
                 vocab_size, layers, dropout, classes):
        super(LSTM_NER_MTL, self).__init__()

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
        self.ne = nn.Sequential(
            nn.Linear(2 * hidden_size, classes),
            nn.LogSoftmax(dim=2))

        self.is_ne = nn.Sequential(
            nn.Linear(2 * hidden_size, 1),
            nn.Sigmoid())

    def forward(self, x):
        # Embedding
        emb = self.embedding(x)

        # RNN
        output, _ = self.rnn(emb)

        # Classifier
        ne = self.ne(output)
        is_ne = self.is_ne(output).squeeze(2)

        # Reshape for loss
        ne = torch.transpose(ne, 1, 2)

        return (ne, is_ne)


class LSTM_NER_Attention(nn.Module):

    def __init__(self, embedding_size, hidden_size,
                 vocab_size, layers, dropout, classes):
        super(LSTM_NER_Attention, self).__init__()

        # Parameters
        self.layers = layers
        self.classes = classes
        self.hidden_size = hidden_size

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # Encoder
        self.encoder = nn.LSTM(embedding_size, hidden_size, layers,
                               dropout=dropout, batch_first=True,
                               bidirectional=True)

        # Attention
        self.attention = SelfAttention(2 * hidden_size, batch_first=True)

        # Decoder
        self.decoder = nn.LSTM(4 * hidden_size, hidden_size, layers,
                               dropout=dropout, batch_first=True)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, classes),
            nn.LogSoftmax(dim=2))

    def forward(self, x):
        # Embedding
        emb = self.embedding(x)

        # Encoder
        encoded, _ = self.encoder(emb)

        # Attention
        represent, attent = self.attention(encoded)

        # reshape attention
        represent = represent.unsqueeze(1).repeat(1, encoded.shape[1], 1)
        concat = torch.cat((encoded, represent), dim=2)

        # Decoder + concatenated attention
        decoded, _ = self.decoder(concat)

        # Classifier
        y = self.classifier(decoded)

        # Reshape for loss
        y = torch.transpose(y, 1, 2)

        return y


class LSTM_NER_Proposal(nn.Module):

    def __init__(self, embedding_size, hidden_size,
                 vocab_size, layers, dropout, classes):
        super(LSTM_NER_Proposal, self).__init__()

        # Parameters
        self.layers = layers
        self.classes = classes
        self.hidden_size = hidden_size

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # Encoder
        self.encoder = nn.LSTM(embedding_size, hidden_size, layers,
                               dropout=dropout, batch_first=True,
                               bidirectional=True)

        # Attention
        self.attention = SelfAttention(2 * hidden_size, batch_first=True)

        # Decoder
        self.decoder = nn.LSTM(4 * hidden_size, hidden_size, layers,
                               dropout=dropout, batch_first=True)

        # Classifiers
        self.ne = nn.Sequential(
            nn.Linear(hidden_size, classes),
            nn.LogSoftmax(dim=2))

        self.is_ne = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())

    def forward(self, x):
        # Embedding
        emb = self.embedding(x)

        # Encoder
        encoded, _ = self.encoder(emb)

        # Attention
        represent, attent = self.attention(encoded)

        # reshape attention
        represent = represent.unsqueeze(1).repeat(1, encoded.shape[1], 1)
        concat = torch.cat((encoded, represent), dim=2)

        # Decoder + concatenated attention
        decoded, _ = self.decoder(concat)

        # Classifier
        ne = self.ne(decoded)
        is_ne = self.is_ne(decoded).squeeze(2)

        # Reshape for loss
        ne = torch.transpose(ne, 1, 2)

        return (ne, is_ne)


class SelfAttention(nn.Module):
    # taken and adapted from
    # https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/4

    def __init__(self, hidden_size, batch_first=False):
        super(SelfAttention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size),
                                        requires_grad=True)

        nn.init.xavier_uniform_(self.att_weights.data)

    def forward(self, inputs):

        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)
                            # (batch_size, hidden_size, 1)
                            )

        attentions = F.softmax(F.relu(weights.squeeze()), dim=1)

        # apply attention weights
        weighted = torch.mul(inputs,
                             attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions
