# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as models


class Resnet_Posture(nn.Module):

    def __init__(self, args):
        super(Resnet_Posture, self).__init__()

        # Parameters
        self.image_shape = args.image_shape

        # Classifier input dimension
        # self.lin_inpt = 512 * (args.image_shape[0] // 32) * \
        #     (args.image_shape[1] // 32)

        self.resnet = models.resnet18(pretrained=True)

        # self.linear = nn.Sequential(nn.Linear(self.lin_inpt,
        #                                       self.lin_inpt // 2),
        #                             nn.ReLU(),
        #                             nn.Linear(self.lin_inpt // 2,
        #                                       10 *
        #                                       self.image_shape[0] *
        #                                       self.image_shape[1]))

    def forward(self, x):
        print(x.shape)
        x1 = self.resnet(x)
        print(x1.shape)
        # x1 = x1.view(-1, self.lin_inpt)
        # x2 = self.linear(x1)
        # y = x2.view([-1, 10,
        #              self.image_shape[0],
        #              self.image_shape[1]])

        return x1
