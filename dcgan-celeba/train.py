#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms

# Import network
from network import *
from imshow import imshow

# Parser arguments
parser = argparse.ArgumentParser(description='PyTorch DCGAN with CelebA')
parser.add_argument('--train-percentage', '--t',
                    type=float, default=.2, metavar='N',
                    help='porcentage of the training set to use (default: .2)')
parser.add_argument('--batch-size', '--b',
                    type=int, default=4, metavar='N',
                    help='input batch size for training (default: 4)')
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
parser.add_argument('--latent_dim', '--ld',
                    type=int, default=2, metavar='N',
                    help='dimension of the latent space (default: 30)')
parser.add_argument('--optimizer', '--o',
                    default='adam', choices=['adam', 'sgd', 'rmsprop'],
                    help='pick a specific optimizer (default: "adam")')
parser.add_argument('--dataset', '--data',
                    default='mnist', choices=['mnist', 'fashion-mnist'],
                    help='pick a specific dataset (default: "mnist")')
parser.add_argument('--plot', '--p',
                    action='store_true',
                    help='plot dataset sample')
parser.add_argument('--no-plot', '--np',
                    dest='plot', action='store_false',
                    help='do not plot dataset sample')
args = parser.parse_args()
print(args)


def train(trainset):

    # Split dataset
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
    if args.plot:
        # get some random training images
        dataiter = iter(train_loader)
        images, _ = dataiter.next()

        grid = torchvision.utils.make_grid(images)
        imshow(grid)
        args.writer.add_image('sample-train', grid)

    # Define optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(args.net.parameters(), lr=1e-3)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(args.net.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(args.net.parameters(), lr=0.01)

    # Loss function
    criterion = elbo_loss_function

    # Set best for minimization
    best = float('inf')

    print('Started Training')
    # loop over the dataset multiple times
    for epoch in range(args.epochs):
        # reset running loss statistics
        train_loss = mse_loss = running_loss = 0.0

        for batch_idx, data in enumerate(train_loader, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, _ = data
            inputs = inputs.to(args.device)

            with autograd.detect_anomaly():
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs, mu, logvar = args.net(inputs)
                loss = criterion(outputs, inputs, mu, logvar)
                loss.backward()
                optimizer.step()

            # update running loss statistics
            train_loss += loss.item()
            running_loss += loss.item()
            mse_loss += F.mse_loss(outputs, inputs)

            # Global step
            global_step = batch_idx + len(train_loader) * epoch

            # Write tensorboard statistics
            args.writer.add_scalar('Train/loss', loss.item(), global_step)
            args.writer.add_scalar('Train/mse', F.mse_loss(outputs, inputs),
                                   global_step)

            # check if current batch had best fitness
            if loss.item() < best:
                best = loss.item()
                update_best(inputs, outputs, loss, global_step)

            # print every args.log_interval of batches
            if batch_idx % args.log_interval == 0:
                print("Train Epoch : {} Batches : {} "
                      "[{}/{} ({:.0f}%)]\tLoss : {:.6f}"
                      "\tError : {:.6f}"
                      .format(epoch, batch_idx,
                              args.batch_size * batch_idx,
                              len(train_loader.dataset),
                              100. * batch_idx / len(train_loader),
                              running_loss / args.log_interval,
                              mse_loss / args.log_interval))

                mse_loss = running_loss = 0.0

                # Add images to tensorboard
                write_images_to_tensorboard(inputs, outputs,
                                            global_step, step=True)

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader)))

    # Add trained model
    args.writer.close()
    print('Finished Training')


def test(testset):
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=args.batch_size,
                                              shuffle=True)
    # Test network
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images = images.to(args.device)

    # Add net to tensorboard
    args.writer.add_graph(args.net, images)

    # Forward through network
    outputs, mu, logvar = args.net(images)

    # Show autoencoder fit and random latent decoded
    if args.plot:
        write_images_to_tensorboard(images, outputs,
                                    global_step=None,
                                    step=False)

    # Plot latent space
    if args.latent_dim == 2 and args.plot:
        plot_latent_space(dataiter, images, labels)


def update_best(inputs, outputs, loss, global_step):
    # Save current state of model
    state = args.net.state_dict()
    torch.save(state, "best/best_{}.pt".format(args.run))

    # Write tensorboard statistics
    args.writer.add_scalar('Best/loss', loss.item(), global_step)
    args.writer.add_scalar('Best/mse', F.mse_loss(outputs, inputs),
                           global_step)

    # Add tensorboard images
    write_images_to_tensorboard(inputs, outputs, global_step,
                                step=False, best=True)


def plot_latent_space(dataiter, images, labels):
    # Plot mesh grid from latent space
    numImgs = 30
    lo, hi = -3., 3.

    # Define mesh grid ticks
    z1 = torch.linspace(lo, hi, numImgs)
    z2 = torch.linspace(lo, hi, numImgs)

    # Create mesh as pair of elements
    z = []
    for idx in range(numImgs):
        for jdx in range(numImgs):
            z.append([z1[idx], z2[jdx]])
    z = torch.tensor(z).to(args.device)

    # Decode elements from latent space
    decoded_z = args.net.decode(z).cpu()

    # print images
    grid = torchvision.utils.make_grid(decoded_z, nrow=numImgs)
    imshow(grid)
    args.writer.add_image('latent-space-grid-decoded', grid)

    # Plot encoded test set into latent space
    numBatches = 500
    for idx in range(numBatches):
        tImages, tLabels = dataiter.next()
        images = torch.cat((images.cpu(), tImages.cpu()), 0)
        labels = torch.cat((labels.cpu(), tLabels.cpu()), 0)

    # encode into latent space
    images = images.cpu()
    encoded_images_loc, _ = args.net.cpu().encode(images)
    encoded_images_loc = encoded_images_loc.cpu().detach().numpy()

    # Scatter plot of latent space
    x = encoded_images_loc[:, 0]
    y = encoded_images_loc[:, 1]

    # Send to tensorboard
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)
    sct = ax.scatter(x, y, c=labels, cmap='jet')
    fig.colorbar(sct)
    args.writer.add_figure('scatter-plot-of-encoded-test-sample', fig)

    # Plot with matplotlib
    plt.figure(figsize=(12, 10))
    plt.scatter(x, y, c=labels, cmap='jet')
    plt.colorbar()
    filename = "imgs/test_into_latent_space_{}.png".format(numBatches)
    plt.savefig(filename)
    plt.show()


def write_images_to_tensorboard(inputs, outputs, global_step,
                                step=False, best=False):
    # Add images to tensorboard
    # Current autoencoder fit
    grid1 = torchvision.utils.make_grid(torch.cat((
        inputs.cpu(), outputs.cpu())), nrow=args.batch_size)

    # Current quality of generated random images
    sample_size = 4 * args.batch_size
    sample = torch.randn(sample_size, args.latent_dim).to(args.device)
    decoded_sample = args.net.decode(sample).cpu()

    # print images
    grid2 = torchvision.utils.make_grid(decoded_sample,
                                        nrow=args.batch_size)

    if step:
        args.writer.add_image('Train/fit', grid1, global_step)
        args.writer.add_image('Train/generated', grid2, global_step)
    elif best:
        args.writer.add_image('Best/fit', grid1, global_step)
        args.writer.add_image('Best/generated', grid2, global_step)
    else:
        args.writer.add_image('encoder-fit', grid1)
        args.writer.add_image('latent-random-sample-decoded', grid2)
        imshow(grid1)
        imshow(grid2)


def create_run_name():
    run = '{}={}'.format('nw', args.network)
    run += '_{}={}'.format('ds', args.dataset)
    run += '_{}={}'.format('ld', args.latent_dim)
    run += '_{}={}'.format('op', args.optimizer)
    run += '_{}={}'.format('ep', args.epochs)
    run += '_{}={}'.format('bs', args.batch_size)
    run += '_{}={}'.format('tp', args.train_percentage)

    return run


def main():
    # Save parameters in string to name the execution
    args.run = create_run_name()

    # Tensorboard summary writer
    args.writer = SummaryWriter('runs/' + args.run)

    # Printing parameters
    torch.set_printoptions(precision=10)
    torch.set_printoptions(edgeitems=5)

    # Set up GPU
    if args.device != 'cpu':
        args.device = torch.device('cuda:0'
                                   if torch.cuda.is_available()
                                   else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(args.device)

    # Create network
    if args.network == 'dcvae':
        net = DCVAE(args.latent_dim)
    elif args.network == 'vae':
        net = VAE(args.latent_dim)

    args.net = net.to(args.device)
    print(args.net)

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST("", train=True,
                                              transform=transform,
                                              download=True)
        testset = torchvision.datasets.MNIST("", train=False,
                                             transform=transform,
                                             download=True)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST("", train=True,
                                                     transform=transform,
                                                     download=True)
        testset = torchvision.datasets.FashionMNIST("", train=False,
                                                    transform=transform,
                                                    download=True)
    # Train network
    train(trainset)

    # Test the trained model
    test(testset)

    # Close tensorboard writer
    args.writer.close()


if __name__ == "__main__":
    main()
