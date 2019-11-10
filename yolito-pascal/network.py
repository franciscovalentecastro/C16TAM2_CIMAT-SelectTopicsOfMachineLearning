# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as models


class loss_yolo():
    def __init__(self, args):
        # Network to train
        self.network = args.network

    def loss(self, outputs, targets):
        batch_size = outputs.shape[0]
        lmbd_coord = 5
        lmbd_noobj = .5

        loss = 0
        for idx in range(batch_size):

            # Get output elements
            c_out = outputs[idx][0]
            y_out = outputs[idx][1]
            x_out = outputs[idx][2]
            w_out = torch.sqrt(outputs[idx][3])
            h_out = torch.sqrt(outputs[idx][4])
            c1_out = outputs[idx][5]
            c2_out = outputs[idx][6]
            c3_out = outputs[idx][7]
            c4_out = outputs[idx][8]

            # Ger target elemtns
            c_trgt = targets[idx][0]
            y_trgt = targets[idx][1]
            x_trgt = targets[idx][2]
            w_trgt = torch.sqrt(targets[idx][3])
            h_trgt = torch.sqrt(targets[idx][4])
            c1_trgt = targets[idx][5]
            c2_trgt = targets[idx][6]
            c3_trgt = targets[idx][7]
            c4_trgt = targets[idx][8]

            # Calculate loss
            for x in range(7):
                for y in range(7):
                    if c_trgt[x, y] == 1.0:
                        loss = lmbd_coord * (x_out[x, y] - x_trgt[x, y]) ** 2
                        loss += lmbd_coord * (y_out[x, y] - y_trgt[x, y]) ** 2
                        loss += lmbd_coord * (w_out[x, y] - w_trgt[x, y]) ** 2
                        loss += lmbd_coord * (h_out[x, y] - h_trgt[x, y]) ** 2
                        loss += (c_out[x, y] - c_trgt[x, y]) ** 2
                        loss += (c1_out[x, y] - c1_trgt[x, y]) ** 2
                        loss += (c2_out[x, y] - c2_trgt[x, y]) ** 2
                        loss += (c3_out[x, y] - c3_trgt[x, y]) ** 2
                        loss += (c4_out[x, y] - c4_trgt[x, y]) ** 2
                    else:
                        loss += lmbd_noobj * (c_out[x, y] - c_trgt[x, y]) ** 2

        return loss.float() / batch_size


class YOLO(nn.Module):

    def __init__(self):
        super(YOLO, self).__init__()

        self.vgg = models.vgg16(pretrained=True)

        self.linear = nn.Sequential(nn.Linear(8192, 9 * 7 * 7),
                                    nn.Sigmoid())

    def forward(self, x):
        x1 = self.vgg.features(x)
        x1 = x1.view(-1, 8192)
        x2 = self.linear(x1)
        y = x2.view([-1, 9, 7, 7])

        return y
