# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as models


class loss_yolo():
    def __init__(self, args):
        # Network to train
        self.device = args.device

    def loss(self, outputs, targets):
        batch_size = outputs.shape[0]
        lmbd_coord = 5
        lmbd_noobj = .5

        # Get output elements
        c_out = outputs[:, 0].reshape(batch_size * 49)
        y_out = outputs[:, 1].reshape(batch_size * 49)
        x_out = outputs[:, 2].reshape(batch_size * 49)
        w_out = torch.sqrt(outputs[:, 3]).reshape(batch_size * 49)
        h_out = torch.sqrt(outputs[:, 4]).reshape(batch_size * 49)
        c1_out = outputs[:, 5].reshape(batch_size * 49)
        c2_out = outputs[:, 6].reshape(batch_size * 49)
        c3_out = outputs[:, 7].reshape(batch_size * 49)
        c4_out = outputs[:, 8].reshape(batch_size * 49)

        # Ger target elemtns
        c_trgt = targets[:, 0].reshape(batch_size * 49)
        y_trgt = targets[:, 1].reshape(batch_size * 49)
        x_trgt = targets[:, 2].reshape(batch_size * 49)
        w_trgt = torch.sqrt(targets[:, 3]).reshape(batch_size * 49)
        h_trgt = torch.sqrt(targets[:, 4]).reshape(batch_size * 49)
        c1_trgt = targets[:, 5].reshape(batch_size * 49)
        c2_trgt = targets[:, 6].reshape(batch_size * 49)
        c3_trgt = targets[:, 7].reshape(batch_size * 49)
        c4_trgt = targets[:, 8].reshape(batch_size * 49)

        # Calculate loss
        obj = c1_trgt + c2_trgt + c3_trgt + c4_trgt
        noobj = torch.ones(batch_size * 49).to(self.device) - obj

        loss = (lmbd_coord * obj * (x_out - x_trgt) ** 2).sum()
        loss += (lmbd_coord * obj * (y_out - y_trgt) ** 2).sum()
        loss += (lmbd_coord * obj * (w_out - w_trgt) ** 2).sum()
        loss += (lmbd_coord * obj * (h_out - h_trgt) ** 2).sum()
        loss += (obj * (c_out - c_trgt) ** 2).sum()
        loss += (obj * (c1_out - c1_trgt) ** 2).sum()
        loss += (obj * (c2_out - c2_trgt) ** 2).sum()
        loss += (obj * (c3_out - c3_trgt) ** 2).sum()
        loss += (obj * (c4_out - c4_trgt) ** 2).sum()
        loss += (lmbd_noobj * noobj * (c_out - c_trgt) ** 2).sum()

        return loss / batch_size


class YOLO(nn.Module):

    def __init__(self):
        super(YOLO, self).__init__()

        self.vgg = models.vgg16(pretrained=True)

        self.linear = nn.Sequential(nn.Linear(8192, 4096),
                                    nn.ReLU(),
                                    nn.Linear(4096, 9 * 7 * 7),
                                    nn.Sigmoid())

    def forward(self, x):
        x1 = self.vgg.features(x)
        x1 = x1.view(-1, 8192)
        x2 = self.linear(x1)
        y = x2.view([-1, 9, 7, 7])

        return y
