#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms

# Import network
from network import *
from imshow import *

# Parser arguments
parser = argparse.ArgumentParser(description='Test PyTorch DCGAN with CelebA')
parser.add_argument('checkpoint',
                    help='path to checkpoint to be restored for inference')
parser.add_argument('--batch-size', '--b',
                    type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 16)')
parser.add_argument('--device', '--d',
                    default='cpu', choices=['cpu', 'cuda'],
                    help='pick device to run the training (defalut: "cpu")')
parser.add_argument('--network', '--n',
                    default='dcgan',
                    choices=['dcgan', 'gan', 'mixed'],
                    help='pick a specific network to train (default: dcgan)')
parser.add_argument('--latent_dim', '--ld',
                    type=int, default=2, metavar='N',
                    help='dimension of the latent space (default: 30)')
parser.add_argument('--filters', '--f',
                    type=int, default=16, metavar='N',
                    help='multiple of number of filters to use (default: 16)')
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


def test(testset):
    # Create dataset loader
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=args.batch_size,
                                              shuffle=True)

    # Show sample of images
    if args.plot:
        # get some random training images
        dataiter = iter(train_loader)
        images, _ = dataiter.next()

        # Image range from (-1,1) to (0,1)
        grid = torchvision.utils.make_grid(0.5 * (images + 1.0))
        imshow(grid)
        args.writer.add_image('sample-test', grid)

    # Restore past checkpoint
    restore_checkpoint()

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
        plot_latent_space(dataiter, images, labels, args)


def restore_checkpoint():
    if args.checkpoint == 'none':
        return

    # Load provided checkpoint
    checkpoint = torch.load(args.checkpoint)
    print('Restored weights from {}.'.format(args.checkpoint))

    # Restore past checkpoint
    args.generator.load_state_dict(checkpoint['generator_state_dict'])
    args.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    args.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    args.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])


def process_checkpoint(lossG, global_step):

    # check if current batch had best generating fitness
    steps_before_best = 2000
    if lossG < args.bestG and global_step > steps_before_best:
        args.bestG = lossG

        # Save best checkpoint
        torch.save({
            'generator_state_dict': args.generator.state_dict(),
            'discriminator_state_dict': args.discriminator.state_dict(),
            'optimizerG_state_dict': args.optimizerG.state_dict(),
            'optimizerD_state_dict': args.optimizerD.state_dict(),
        }, "checkpoint/best_{}.pt".format(args.run))

        # Write tensorboard statistics
        args.writer.add_scalar('Best/lossG', lossG, global_step)

        # Save best generation image
        write_images_to_tensorboard(global_step, best=True)

    # Save current checkpoint
    torch.save({
        'generator_state_dict': args.generator.state_dict(),
        'discriminator_state_dict': args.discriminator.state_dict(),
        'optimizerG_state_dict': args.optimizerG.state_dict(),
        'optimizerD_state_dict': args.optimizerD.state_dict(),
    }, "checkpoint/last_{}.pt".format(args.run))


def write_images_to_tensorboard(global_step, step=False, best=False):
    # Current quality of generated random images
    sample_size = 16
    sample = torch.randn(sample_size, args.latent_dim).to(args.device)

    # Image range from (-1,1) to (0,1)
    generated_sample = 0.5 * (args.generator(sample).cpu() + 1.0)

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
    run += '_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))

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

    # Set image size
    args.image_size = 64

    # Set dataset transform
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Load dataset
    if args.dataset == 'celeba':
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        testset = torchvision.datasets.ImageFolder(root='CelebA/',
                                                   transform=transform)
    elif args.dataset == 'mnist':
        testset = torchvision.datasets.MNIST("", train=False,
                                             transform=transform,
                                             download=True)

    elif args.dataset == 'fashion-mnist':
        testset = torchvision.datasets.FashionMNIST("", train=False,
                                                    transform=transform,
                                                    download=True)

    # Determine image shape
    tImage, _ = testset[0]
    args.image_shape = tImage.shape

    # Create network
    if args.network == 'dcgan':
        discriminator = ConvolutionalDiscriminator(args.image_shape,
                                                   args.filters)
        generator = ConvolutionalGenerator(args.latent_dim,
                                           args.image_shape,
                                           args.filters)
    elif args.network == 'gan':
        discriminator = Discriminator(args.image_shape)
        generator = Generator(args.latent_dim, args.image_shape)
    elif args.network == 'mixed':
        discriminator = Discriminator(args.image_shape)
        generator = ConvolutionalGenerator(args.latent_dim,
                                           args.image_shape,
                                           args.filters)

    # Send networks to device
    args.discriminator = discriminator.to(args.device)
    args.generator = generator.to(args.device)

    print(args.discriminator)
    print(args.generator)

    # Test the trained model
    test(testset)

    # Close tensorboard writer
    args.writer.close()


if __name__ == "__main__":
    main()
