# -*- coding: utf-8 -*-
import torch.nn as nn
import torchvision.models as models


class Resnet_Posture(nn.Module):

    def __init__(self, args):
        super(Resnet_Posture, self).__init__()

        # Parameters
        self.image_shape = args.image_shape

        # Resnet
        self.resnet = models.resnet18(pretrained=True)

        # Deconv
        self.deconv1 = self.deconv_block(512, 128, kernel_size=4, stride=4)
        self.deconv2 = self.deconv_block(128, 64, kernel_size=4, stride=4)
        self.deconv3 = self.deconv_block(64, 17, kernel_size=2, stride=2)

    def deconv_block(self, inpt, outpt, **kwargs):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=inpt, out_channels=outpt, **kwargs),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print('{} {}'.format(1, x.shape))
        x = self.resnet.conv1(x)
        # print('{} {}'.format(2, x.shape))
        x = self.resnet.bn1(x)
        # print('{} {}'.format(3, x.shape))
        x = self.resnet.relu(x)
        # print('{} {}'.format(4, x.shape))
        x = self.resnet.maxpool(x)
        # print('{} {}'.format(5, x.shape))
        x = self.resnet.layer1(x)
        # print('{} {}'.format(6, x.shape))
        x = self.resnet.layer2(x)
        # print('{} {}'.format(7, x.shape))
        x = self.resnet.layer3(x)
        # print('{} {}'.format(8, x.shape))
        x = self.resnet.layer4(x)
        # print('{} {}'.format(9, x.shape))
        x = self.deconv1(x)
        # print('{} {}'.format(10, x.shape))
        x = self.deconv2(x)
        # print('{} {}'.format(11, x.shape))
        x = self.deconv3(x)
        # print('{} {}'.format(12, x.shape))

        return x
