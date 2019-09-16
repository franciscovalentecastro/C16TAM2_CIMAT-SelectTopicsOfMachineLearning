#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import torch
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

# Import network
from network import *
from imshow import imshow

# Parser arguments
parser = argparse.ArgumentParser(description='PyTorch DCVAE with MNIST')
parser.add_argument('--train-percentage', '--t',
                    type=float, default=.2, metavar='N',
                    help='porcentage of the training set to use (default: .2)')
parser.add_argument('--batch-size', '--b',
                    type=int, default=4, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--log-interval', '--li',
                    type=int, default=100, metavar='N',
                    help='how many batches to wait' +
                         'before logging training status')
parser.add_argument('--epochs', '--e',
                    type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--device', '--d',
                    default='cpu', choices=['cpu', 'cuda'],
                    help='pick device to run the training (defalut: "cpu")')
parser.add_argument('--network', '--n',
                    default='dcvae', choices=['dcvae', 'vae'],
                    help='pick a specific network to train (default: dcvae)')
parser.add_argument('--optimizer', '--o',
                    default='adam', choices=['adam', 'sgd', 'rmsprop'],
                    help='pick a specific optimizer (default: "adam")')
args = parser.parse_args()
print(args)


def train():
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST("", train=True,
                                          transform=transform,
                                          download=True)

    train_size = int(args.train_percentage * len(trainset))
    test_size = len(trainset) - train_size
    train_dataset, test_dataset \
        = torch.utils.data.random_split(trainset, [train_size, test_size])

    # Dataset information
    print('train dataset : {} elements'.format(len(train_dataset)))

    # Create dataset loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    # Show sample of images
    # get some random training images
    dataiter = iter(train_loader)
    images, _ = dataiter.next()

    imshow(torchvision.utils.make_grid(images))

    # Define optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(args.net.parameters(), lr=1e-3)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(args.net.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(args.net.parameters(), lr=0.01)
    else:
        optimizer = optim.Adam(args.net.parameters(), lr=1e-3)

    print('Started Training')
    # loop over the dataset multiple times
    for epoch in range(args.epochs):

        train_loss = 0.0
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(args.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, mu, logvar = args.net(inputs)
            loss = elbo_loss_function(outputs, inputs, mu, logvar)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()
            running_loss += loss.item()

            # print every number_of_mini_batches
            if batch_idx % args.log_interval == 0:
                print("Train Epoch : {} Batches : {} "
                      "[{}/{} ({:.0f}%)]\tLoss : {:.6f}"
                      .format(epoch, batch_idx,
                              args.batch_size * batch_idx,
                              len(train_loader.dataset),
                              100. * batch_idx / len(train_loader),
                              running_loss / args.log_interval))

                running_loss = 0.0

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    print('Finished Training')


def test():
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST("", train=False,
                                         transform=transform,
                                         download=True)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=True)
    # Test network
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    if torch.cuda.is_available():
        images_cuda = images.to(args.device)
        outputs, mu, logvar = args.net(images_cuda)
    else:
        outputs, mu, logvar = args.net(images)

    outputs_cpu = outputs.cpu()

    # print images
    imshow(torchvision.utils.make_grid(
           torch.cat((images, outputs_cpu)), nrow=args.batch_size))

    # Sample normal distribution
    sample = torch.randn(16, 2).to(args.device)
    sample = args.net.decode(sample).cpu()

    # print images
    imshow(torchvision.utils.make_grid(sample, nrow=args.batch_size))


def main():
    # Printing parameters
    torch.set_printoptions(precision=10)
    torch.set_printoptions(edgeitems=5)

    # Set up GPU
    if args.device is not 'cpu':
        args.device = torch.device('cuda:0'
                                   if torch.cuda.is_available()
                                   else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(args.device)

    # Create network
    if args.network == 'dcvae':
        net = DCVAE()
    elif args.network == "vae":
        net = VAE()
    else:
        net = DCVAE()

    args.net = net.to(args.device)
    print(args.net)

    # Train network
    train()

    # Test the trained model
    test()


if __name__ == "__main__":
    main()
