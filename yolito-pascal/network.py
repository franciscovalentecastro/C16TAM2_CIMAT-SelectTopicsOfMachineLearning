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
            c_out = outputs[idx][0].view(49)
            y_out = outputs[idx][1].view(49)
            x_out = outputs[idx][2].view(49)
            w_out = torch.sqrt(outputs[idx][3]).view(49)
            h_out = torch.sqrt(outputs[idx][4]).view(49)
            c1_out = outputs[idx][5].view(49)
            c2_out = outputs[idx][6].view(49)
            c3_out = outputs[idx][7].view(49)
            c4_out = outputs[idx][8].view(49)

            # Ger target elemtns
            c_trgt = targets[idx][0].view(49)
            y_trgt = targets[idx][1].view(49)
            x_trgt = targets[idx][2].view(49)
            w_trgt = torch.sqrt(targets[idx][3]).view(49)
            h_trgt = torch.sqrt(targets[idx][4]).view(49)
            c1_trgt = targets[idx][5].view(49)
            c2_trgt = targets[idx][6].view(49)
            c3_trgt = targets[idx][7].view(49)
            c4_trgt = targets[idx][8].view(49)

            # Calculate loss
            obj = c1_trgt + c2_trgt + c3_trgt + c4_trgt
            noobj = torch.ones(49) - obj

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

        self.linear = nn.Sequential(nn.Linear(8192, 9 * 7 * 7),
                                    nn.Sigmoid())

    def forward(self, x):
        x1 = self.vgg.features(x)
        x1 = x1.view(-1, 8192)
        x2 = self.linear(x1)
        y = x2.view([-1, 9, 7, 7])

        return y
