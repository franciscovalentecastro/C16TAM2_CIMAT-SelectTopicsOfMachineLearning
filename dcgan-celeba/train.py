#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
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
                    type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
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
                    default='dcgan', choices=['dcgan', 'gan', 'mixed'],
                    help='pick a specific network to train (default: dcgan)')
parser.add_argument('--latent_dim', '--ld',
                    type=int, default=2, metavar='N',
                    help='dimension of the latent space (default: 30)')
parser.add_argument('--optimizer', '--o',
                    default='adam', choices=['adam', 'sgd', 'rmsprop'],
                    help='pick a specific optimizer (default: "adam")')
parser.add_argument('--dataset', '--data',
                    default='celeba',
                    choices=['celeba', 'mnist', 'fashion-mnist'],
                    help='pick a specific dataset (default: "celeba")')
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
                                               shuffle=True,
                                               drop_last=True)
    args.dataset_size = len(train_loader.dataset)
    args.dataloader_size = len(train_loader)

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
        optimizerD = optim.Adam(args.discriminator.parameters(),
                                lr=1e-3, betas=(0.5, 0.999))
        optimizerG = optim.Adam(args.generator.parameters(),
                                lr=1e-3, betas=(0.5, 0.999))
    elif args.optimizer == 'sgd':
        optimizerD = optim.SGD(args.discriminator.parameters(),
                               lr=0.01, momentum=0.9)
        optimizerG = optim.SGD(args.generator.parameters(),
                               lr=0.01, momentum=0.9)

    # Set best for minimization
    args.bestD = float('inf')
    args.bestG = float('inf')

    print('Started Training')
    # loop over the dataset multiple times
    for epoch in range(args.epochs):

        # reset running loss statistics
        args.train_lossD = args.running_lossD = 0.0
        args.train_lossG = args.running_lossG = 0.0

        for batch_idx, data in enumerate(train_loader, 1):

            # Get data
            inputs, _ = data
            inputs = inputs.to(args.device)

            # Update discriminator
            outputsR, outputsF, lossD = train_update_net('discriminator',
                                                         inputs, optimizerD)

            # update running loss statistics
            args.train_lossD += lossD
            args.running_lossD += lossD

            # Update Generator
            outputsG, lossG = train_update_net('generator',
                                               [], optimizerG)

            # Global step
            global_step = batch_idx + len(train_loader) * epoch

            # update running loss statistics
            args.train_lossG += lossG
            args.running_lossG += lossG

            # Write tensorboard statistics
            args.writer.add_scalar('Train/lossD', lossD, global_step)
            args.writer.add_scalar('Train/lossG', lossG, global_step)

            # print every args.log_interval of batches
            if batch_idx % args.log_interval == 0:
                print('Train Epoch : {} Batches : {} [{}/{} ({:.0f}%)]'
                      '\tLossD : {:.8f} \tLossG : {:.8f}'
                      .format(epoch, batch_idx,
                              args.batch_size * batch_idx,
                              args.dataset_size,
                              100. * batch_idx / args.dataloader_size,
                              args.running_lossD / args.log_interval,
                              args.running_lossG / args.log_interval))

                args.running_lossG = args.running_lossD = 0.0

                # Add images to tensorboard
                write_images_to_tensorboard(global_step, step=True)

            # check if current batch had best fitness
            # if lossD < bestD:
            #     bestD = lossD
            # update_best(inputs, outputs, lossD, global_step)

        print('====> Epoch: {} '
              'Average lossD: {:.4f} '
              'Average lossG: {:.4f}'
              .format(epoch,
                      args.train_lossD / len(train_loader),
                      args.train_lossG / len(train_loader)))

    # Add trained model
    print('Finished Training')


def train_update_net(network, inputs, optimizer):

    if network == 'discriminator':

        # Calculate gradients and update
        with autograd.detect_anomaly():
            # zero the parameter gradients
            optimizer.zero_grad()

            # Update discriminator with real data
            labels_real = torch.ones(args.batch_size).to(args.device)

            # forward + backward real
            outputs_real = args.discriminator(inputs)
            loss_real = F.binary_cross_entropy(outputs_real, labels_real)
            loss_real.backward()

            # update weights
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            # Update discriminator with fake data
            labels_fake = torch.zeros(args.batch_size).to(args.device)

            # Get noise sample from latent space
            latent = torch.randn((args.batch_size, args.latent_dim)) \
                .to(args.device)

            # Generate fake data from generator
            fake = args.generator(latent)

            # forward + backward fake
            outputs_fake = args.discriminator(fake)
            loss_fake = F.binary_cross_entropy(outputs_fake, labels_fake)
            loss_fake.backward()

            # update weights
            optimizer.step()

            # Calculate loss
            loss = loss_real + loss_fake

        return outputs_real, outputs_fake, loss.item()

    elif network == 'generator':

        # Calculate gradients and update
        with autograd.detect_anomaly():
            # zero the parameter gradients
            optimizer.zero_grad()

            # Update discriminator with fake data
            labels_mistake = torch.ones(args.batch_size).to(args.device)

            # Get noise sample from latent space
            latent = torch.randn((args.batch_size, args.latent_dim)) \
                .to(args.device)

            # Generate fake data from generator
            fake = args.generator(latent)

            # forward + backward + step
            outputs = args.discriminator(fake)
            loss = F.binary_cross_entropy(outputs, labels_mistake)
            loss.backward()
            optimizer.step()

        return outputs, loss.item()


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


def write_images_to_tensorboard(global_step, step=False, best=False):
    # Current quality of generated random images
    sample_size = 16
    sample = torch.randn(sample_size, args.latent_dim).to(args.device)
    generated_sample = args.generator(sample).cpu()

    # print images
    grid = torchvision.utils.make_grid(generated_sample, nrow=4)

    if step:
        args.writer.add_image('Train/generated', grid, global_step)
    elif best:
        args.writer.add_image('Best/generated', grid, global_step)
    else:
        args.writer.add_image('latent-random-sample-decoded', grid)
        imshow(grid)


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

    # Remove previous run folder
    shutil.rmtree('runs/' + args.run, ignore_errors=True)

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

    # Number of steps to train discriminator
    args.discriminator_train_steps = 1

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(args.device)

    # Load dataset
    if args.dataset == 'celeba':
        args.image_size = 64
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.ImageFolder(root='CelebA/',
                                                    transform=transform)
    elif args.dataset == 'mnist':
        args.image_size = 32
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.MNIST("", train=True,
                                              transform=transform,
                                              download=True)
        testset = torchvision.datasets.MNIST("", train=False,
                                             transform=transform,
                                             download=True)
    elif args.dataset == 'fashion-mnist':
        args.image_size = 32
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.FashionMNIST("", train=True,
                                                     transform=transform,
                                                     download=True)
        testset = torchvision.datasets.FashionMNIST("", train=False,
                                                    transform=transform,
                                                    download=True)

    # Determine image shape
    tImage, _ = trainset[0]
    args.image_shape = tImage.shape

    # Create network
    if args.network == 'dcgan':
        discriminator = ConvolutionalDiscriminator(args.image_shape)
        generator = ConvolutionalGenerator(args.latent_dim, args.image_shape)
    elif args.network == 'gan':
        discriminator = Discriminator(args.image_shape)
        generator = Generator(args.latent_dim, args.image_shape)
    elif args.network == 'mixed':
        discriminator = Discriminator(args.image_shape)
        generator = ConvolutionalGenerator(args.latent_dim, args.image_shape)

    # Send networks to device
    args.discriminator = discriminator.to(args.device)
    args.generator = generator.to(args.device)

    print(args.discriminator)
    print(args.generator)

    # Train network
    train(trainset)

    # Test the trained model
    # test(testset)

    # Close tensorboard writer
    args.writer.close()


if __name__ == "__main__":
    main()