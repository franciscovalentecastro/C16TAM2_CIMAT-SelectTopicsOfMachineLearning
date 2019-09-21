import torch
import torch.nn as nn
import torch.nn.functional as F


class DCGAN(nn.Module):

    def __init__(self, latent_dim):
        super(DCGAN, self).__init__()

        # Encoder
        self.conv_block1 = self.down_conv_block(1, 64)
        self.conv_block2 = self.down_conv_block(64, 128)

        self.linear1 = nn.Linear(128 * 7 * 7, 1024)

        self.z_mu = nn.Linear(1024, latent_dim)
        self.z_log_sigma2 = nn.Linear(1024, latent_dim)

        # Decoder
        self.linear2 = nn.Linear(latent_dim, 1024)
        self.linear3 = nn.Linear(1024, 128 * 7 * 7)

        self.conv_block3 = self.up_conv_block(128, 64)
        self.conv_block4 = self.up_conv_block(64, 1)

    def down_conv_block(self, in_chan, out_chan, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=3, stride=2, padding=1, **kwargs),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_chan, out_channels=out_chan,
                      kernel_size=3, stride=1, padding=1, **kwargs),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU()
        )

    def up_conv_block(self, in_chan, out_chan, padding=1, **kwargs):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_chan, out_channels=out_chan,
                               kernel_size=4, stride=2, padding=1, **kwargs),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_chan, out_channels=out_chan,
                      kernel_size=3, stride=1, padding=1, **kwargs),
            nn.BatchNorm2d(out_chan),
            nn.Sigmoid()
        )

    def encode(self, x):
        h1 = self.conv_block1(x)
        h2 = self.conv_block2(h1)
        h3 = F.relu(self.linear1(h2.view(-1, 128 * 7 * 7)))

        return self.z_mu(h3), self.z_log_sigma2(h3)

    def sample(self, z_mu, z_log_sigma2):
        z_std = torch.exp(0.5 * z_log_sigma2)
        epsilon = torch.randn_like(z_std)

        return z_mu + epsilon * z_std

    def decode(self, z):
        h1 = F.relu(self.linear2(z))
        h2 = F.relu(self.linear3(h1))
        h3 = self.conv_block3(h2.view(-1, 128, 7, 7))
        h4 = self.conv_block4(h3)

        return h4

    def forward(self, x):
        z_mu, z_log_sigma2 = self.encode(x)
        z = self.sample(z_mu, z_log_sigma2)

        return self.decode(z), z_mu, z_log_sigma2


class GAN(nn.Module):

    def __init__(self, latent_dim):
        super(GAN, self).__init__()

        # Encoder
        self.linear1 = nn.Linear(784, 256)

        self.z_mu = nn.Linear(256, latent_dim)
        self.z_log_sigma2 = nn.Linear(256, latent_dim)

        # Decoder
        self.linear2 = nn.Linear(latent_dim, 256)
        self.linear3 = nn.Linear(256, 784)

    def encode(self, x):
        h1 = F.relu(self.linear1(x.view(-1, 784)))

        return self.z_mu(h1), self.z_log_sigma2(h1)

    def sample(self, z_mu, z_log_sigma2):
        z_std = torch.exp(0.5 * z_log_sigma2)
        epsilon = torch.randn_like(z_std)

        return z_mu + epsilon * z_std

    def decode(self, z):
        h1 = F.relu(self.linear2(z))
        h2 = torch.sigmoid(self.linear3(h1))

        return h2.view(-1, 1, 28, 28)

    def forward(self, x):
        z_mu, z_log_sigma2 = self.encode(x)
        z = self.sample(z_mu, z_log_sigma2)

        return self.decode(z), z_mu, z_log_sigma2


def elbo_loss_function(decoded_x, x, z_mu, z_log_sigma2):
    # log P(x|z) per element in batch
    LOGP = F.binary_cross_entropy(decoded_x.view(-1, 784),
                                  x.view(-1, 784),
                                  reduction='none')
    LOGP = torch.sum(LOGP, dim=1)

    # DKL(Q(z|x)||P(z)) per element in batch
    DKL = 0.5 * torch.sum(z_log_sigma2.exp() +
                          z_mu.pow(2) -
                          1.0 -
                          z_log_sigma2,
                          dim=1)

    # Average loss in batch
    return torch.mean(LOGP + DKL)
