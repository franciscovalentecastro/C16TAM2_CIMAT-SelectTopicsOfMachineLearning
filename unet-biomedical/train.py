import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# Import utilities
from unet import UNet
from imshow import imshow
import datasets


def main():

    if len(sys.argv) > 5:
        # Get console parameters
        train_percentage = float(sys.argv[1])
        train_batch = int(sys.argv[2])
        test_batch = int(sys.argv[3])
        number_of_epochs = int(sys.argv[4])
        number_of_mini_batches = int(sys.argv[5])
    else:
        print("Not enough parameters")
        return

    # Printing parameters
    torch.set_printoptions(precision=10)
    torch.set_printoptions(edgeitems=32)

    # Set up GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print("Device : ", device)

    # Load dataset
    full_dataset = datasets.ISBI2012Dataset('./dataset/train-volume.tif',
                                            './dataset/train-labels.tif')

    # Divide into Train and Test
    train_size = int(train_percentage * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset \
        = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # Dataset information
    print("train_dataset : ", len(train_dataset))
    print("test_dataset : ", len(test_dataset))

    # Create dataset loaders
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=train_batch,
                                              shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=test_batch,
                                             shuffle=True, num_workers=0)

    # Show sample of images
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(
        torch.cat((images, labels)), nrow=train_batch))

    print('Started Training')

    # Create network
    net = UNet()
    net.to(device)
    # print(net)

    # Define loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train network - loop over the dataset multiple times
    for epoch in range(number_of_epochs):

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
            outputs = net(inputs)
            # cprint(outputs.size(),inputs.size(),labels.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # print every number_of_mini_batches
            if i % number_of_mini_batches == number_of_mini_batches - 1:
                print('[%d, %5d] loss: %.8f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # Test network and predict
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    if torch.cuda.is_available():
        images_cuda = images.cuda()
        labels_cuda = labels.cuda()

    # print images
    imshow(torchvision.utils.make_grid(
        torch.cat((images, labels)), nrow=test_batch))

    outputs = net(images_cuda)

    predicted = ((outputs > .5).type(torch.float) - .5) * 2
    predicted_cpu = predicted.cpu()
    imshow(torchvision.utils.make_grid(
        torch.cat((images, labels, predicted_cpu)), nrow=test_batch))

    # Calculate network accuracy on Test dataset
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            if torch.cuda.is_available():
                images_cuda = images.cuda()
                labels_cuda = labels.cuda()

                outputs = net(images_cuda)
            else:
                outputs = net(images)

            predicted = ((outputs > .5).type(torch.float) - .5) * 2
            predicted_cpu = predicted.cpu()

            # print(images.size(),labels.size(),predicted.size())
            # print(predicted)
            # print(labels)
            # print("total : ",labels.size(0) * labels.size(1) * labels.size(2) * labels.size(3))
            # print("correct : ",(predicted.type(torch.long) == labels_cuda.type(torch.long)).sum().item())
            # print("label lit : ", (labels == 1).sum().item())
            # print("predicted lit : ", (predicted == 1).sum().item())
            # imshow(torchvision.utils.make_grid(torch.cat((images,labels,predicted_cpu)),nrow = test_batch))

            correct += (predicted.type(torch.long) ==
                        labels_cuda.type(torch.long)).sum().item()
            total += labels.size(0) \
                * labels.size(1) \
                * labels.size(2) \
                * labels.size(3) \

    print('Accuracy of the network on the %d test images: %d %%' % (
        len(test_dataset), 100 * correct / total))


if __name__ == "__main__":
    main()
