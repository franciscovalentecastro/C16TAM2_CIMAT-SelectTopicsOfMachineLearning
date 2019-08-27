import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Level 1
        self.left_conv_block1 = self.conv_block(1, 64,
                                                kernel_size=3, padding=1)
        self.right_conv_block = self.conv_block(128, 64,
                                                kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Level 2
        self.left_conv_block2 = self.conv_block(64, 128,
                                                kernel_size=3, padding=1)
        self.right_conv_block2 = self.conv_block(128, 64,
                                                 kernel_size=3, padding=1)

        self.tconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Level 3
        self.left_conv_block3 = self.conv_block(128, 256,
                                                kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Level 4
        self.left_conv_block3 = self.conv_block(256, 512,
                                                kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Level 5 (BottleNeck)
        self.left_conv_block3 = self.conv_block(512, 1024,
                                                kernel_size=3, padding=1)

        # To output
        self.conv5 = nn.Conv2d( in_channels =64, out_channels =  1, 
                                kernel_size = 1, padding = 0)

    def conv_block(in_chan, out_chan, *args, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      *args, **kwargs),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_chan, out_channels=out_chan,
                      *args, **kwargs),
            nn.ReLU()
        )

    def forward(self, x):
        # Level 1
        x1 = self.left_conv_block1(x)
        x2 = self.pool1(x1)

        # Level 2
        x3 = F.relu(self.conv7(F.relu(self.conv6(x2))))
        x4 = self.tconv1(x3)

        # Concatenate
        x5 = torch.cat((x1,x4),1)       

        # Level 1
        x6 = self.conv5(F.relu(self.conv4(F.relu(self.conv3(x5)))))

        # print(x.size())
        # print(x1.size())
        # print(x2.size())
        # print(x3.size())
        # print(x4.size())
        # print(x5.size())
        # print(x6.size())

        return x6

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
