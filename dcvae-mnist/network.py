import torch
import torch.nn as nn
import torch.nn.functional as F


class DCVAE(nn.Module):

    def __init__(self):
        super(DCVAE, self).__init__()

        # Encoder
        self.conv_block1 = self.conv_block(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(8 * 14 * 14, 64)

        self.z_mu = nn.Linear(64, 2)
        self.z_log_sigma2 = nn.Linear(64, 2)

        # Decoder
        self.linear2 = nn.Linear(2, 64)
        self.linear3 = nn.Linear(64, 8 * 14 * 14)

        self.tconv1 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)
        self.conv_block2 = self.conv_block(8, 1, kernel_size=3, padding=1)

    def conv_block(self, in_chan, out_chan, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan, **kwargs),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_chan, out_channels=out_chan, **kwargs),
            nn.ReLU()
        )

    def encode(self, x):
        h1 = self.conv_block1(x)
        h2 = self.pool1(h1)
        h3 = F.relu(self.linear1(h2.view(-1, 8 * 14 * 14)))

        return self.z_mu(h3), self.z_log_sigma2(h3)

    def sample(self, z_mu, z_log_sigma2):
        z_std = torch.exp(0.5 * z_log_sigma2)
        epsilon = torch.randn_like(z_std)

        return z_mu + epsilon * z_std

    def decode(self, z):
        h1 = F.relu(self.linear2(z))
        h2 = F.relu(self.linear3(h1))
        h3 = self.tconv1(h2.view(-1, 8, 14, 14))
        h4 = torch.sigmoid(self.conv_block2(h3))

        return h4

    def forward(self, x):
        z_mu, z_log_sigma2 = self.encode(x)
        z = self.sample(z_mu, z_log_sigma2)
        return self.decode(z), z_mu, z_log_sigma2

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.linear1 = nn.Linear(784, 256)

        self.z_mu = nn.Linear(256, 2)
        self.z_log_sigma2 = nn.Linear(256, 2)

        # Decoder
        self.linear2 = nn.Linear(2, 256)
        self.linear3 = nn.Linear(256, 784)

    def conv_block(self, in_chan, out_chan, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan, **kwargs),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_chan, out_channels=out_chan, **kwargs),
            nn.ReLU()
        )

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


# Reconstruction + KL divergence losses summed over all elements and batch
def elbo_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784),
                                 x.view(-1, 784),
                                 reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = 0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1.0 - logvar)

    return BCE + KLD
