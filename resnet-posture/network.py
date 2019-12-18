# -*- coding: utf-8 -*-
import torch.nn as nn
import torchvision.models as models


class Resnet_Posture(nn.Module):

    def __init__(self, args):
        super(Resnet_Posture, self).__init__()

        # Parameters
        self.image_shape = args.image_shape

        # Resnet
        self.resnet = models.resnet50(pretrained=True)

        # Deconv
        self.deconv1 = self.deconv_block(2048, 512, kernel_size=4, stride=4)
        self.deconv2 = self.deconv_block(512, 128, kernel_size=4, stride=4)
        self.deconv3 = self.deconv_block(128, 17, kernel_size=2, stride=2)

    def deconv_block(self, inpt, outpt, **kwargs):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=inpt, out_channels=outpt, **kwargs),
            nn.BatchNorm2d(outpt),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)

        return x
