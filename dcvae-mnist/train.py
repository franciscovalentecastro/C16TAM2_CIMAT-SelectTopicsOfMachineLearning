#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import torch
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

# Import network
from network import DCVAE, elbo_loss_function
from imshow import imshow


def main():
    # Printing parameters
    torch.set_printoptions(precision=10)
    torch.set_printoptions(edgeitems=5)

    if len(sys.argv) > 3:
        train_percentage = float(sys.argv[1])
        train_batch = int(sys.argv[2])
        test_batch = int(sys.argv[3])
    else:
        print("Not enough parameters")
        return

    # Set up GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.MNIST("", train=True,
                                          transform=transform,
                                          download=True)
    # testset = torchvision.datasets.MNIST("", train=False,
    #                                      transform=transform,
    #                                      download=True)

    # Split into Train and Test
    train_size = int(train_percentage * len(trainset))
    test_size = len(trainset) - train_size
    train_dataset, test_dataset \
        = torch.utils.data.random_split(trainset, [train_size, test_size])

    # Create dataset loader
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=train_batch,
                                              shuffle=True)

    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=test_batch,
                                             shuffle=True)

    # Dataset information
    print("train_dataset : ", len(train_dataset))
    print("test_dataset : ", len(test_dataset))

    # Show sample of images
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%d' % labels[j] for j in range(4)))

    # Create network
    net = DCVAE()
    net.to(device)
    print(net)

    # Define loss function and optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train network
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, mu, logvar = net(inputs)
            loss = elbo_loss_function(outputs, inputs, mu, logvar)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # Test network and predict
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()

        # Test network and predict
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    if torch.cuda.is_available():
        images_cuda = images.cuda()
        outputs, mu, logvar = net(images_cuda)
    else:
        outputs, mu, logvar = net(images)

    outputs_cpu = outputs.cpu()

    # print images
    imshow(torchvision.utils.make_grid(
           torch.cat((images, outputs_cpu)), nrow=test_batch))

    # Sample normal distribution
    sample = torch.randn(16, 2).to(device)
    sample = net.decode(sample).cpu()

    # print images
    imshow(torchvision.utils.make_grid(sample, nrow=test_batch))


if __name__ == "__main__":
    main()
