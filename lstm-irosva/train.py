# -*- coding: utf-8 -*-
import argparse
from datetime import datetime

import torch
import torch.optim as optim
import torch.autograd as autograd

from torchtext.data import *
from torch.utils.tensorboard import SummaryWriter

# Import network
from network import *
from utils import *

# Parser arguments
parser = argparse.ArgumentParser(description='Train PyTorch LSTM '
                                             'Irony Classifier')
parser.add_argument('--train-percentage', '--t',
                    type=float, default=.2, metavar='N',
                    help='porcentage of the training set to use (default: .2)')
parser.add_argument('--batch-size', '--b',
                    type=int, default=8, metavar='N',
                    help='input batch size for training (default: 8)')
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
                    default='lstm',
                    choices=['lstm'],
                    help='pick a specific network to train (default: lstm)')
parser.add_argument('--embedding', '--emb',
                    type=int, default=10, metavar='N',
                    help='dimension of embedding (default: 10)')
parser.add_argument('--hidden', '--h',
                    type=int, default=10, metavar='N',
                    help='dimension of hidden state (default: 10)')
parser.add_argument('--layers', '--ly',
                    type=int, default=2, metavar='N',
                    help='dimension of embedding (default: 2)')
parser.add_argument('--dropout', '--drop',
                    type=float, default=.2, metavar='N',
                    help='dropout percentage in lstm layers (default: .2)')
parser.add_argument('--vocabulary', '--v',
                    type=int, default=5000, metavar='N',
                    help='size of vocabulary (default: 5000)')
parser.add_argument('--optimizer', '--o',
                    default='adam', choices=['adam', 'sgd'],
                    help='pick a specific optimizer (default: "sgd")')
parser.add_argument('--dataset', '--data',
                    default='irosva-mx',
                    choices=['irosva-mx', 'irosva-es', 'irosva-cu'],
                    help='pick a specific dataset (default: "irosva-mx")')
parser.add_argument('--checkpoint', '--check',
                    default='none',
                    help='path to checkpoint to be restored')
parser.add_argument('--plot', '--p',
                    action='store_true',
                    help='plot dataset sample')
parser.add_argument('--summary', '--sm',
                    action='store_true',
                    help='show summary of model')
args = parser.parse_args()
print(args)


def predict(outputs, targets):
    targets = targets.type(torch.long)
    predicted = (outputs > .5).type(torch.long)
    correct = (predicted == targets).sum().item()
    args.train_acc += correct / args.batch_size

    return (predicted, correct)


def batch_status(batch_idx, epoch, train_loader, loss,
                 validationset, irony, topic):
    # Unpack information
    output_irony, target_irony, loss_irony = irony
    output_topic, target_topic, loss_topic = topic

    # Global step
    global_step = batch_idx + len(train_loader) * epoch

    # update running loss statistics
    args.running_loss += loss.item()
    args.train_loss += loss.item()

    # predict
    (predicted, correct) = predict(output_irony, target_irony)

    # Write tensorboard statistics
    args.writer.add_scalar('Train/loss', loss.item(), global_step)
    args.writer.add_scalar('Train/loss_irony', loss_irony.item(), global_step)
    args.writer.add_scalar('Train/loss_topic', loss_topic.item(), global_step)

    # print every args.log_interval of batches
    if global_step % args.log_interval == 0:
        print('Train Epoch : {} Batches : {} [{}/{} ({:.0f}%)]'
              '\tAvg_Loss : {:.4f} Acc_Irony : {:.2f}%\n'
              '\t\tLoss : {:.4f} Loss_Irony : {:.4f} Loss_Topic : {:.4f}'
              .format(epoch, batch_idx,
                      args.batch_size * batch_idx,
                      args.dataset_size,
                      100. * batch_idx / args.dataloader_size,
                      args.running_loss / args.log_interval,
                      100. * correct / args.batch_size,
                      loss, loss_irony, loss_topic))

        args.running_loss = 0.0

        # validate
        validate(validationset, log_info=True, global_step=global_step)

        # Process current checkpoint
        process_checkpoint(loss.item(), target_irony,
                           output_irony, global_step)


def train(trainset, validationset):
    # Create dataset loader
    train_loader = BucketIterator(trainset,
                                  batch_size=args.batch_size,
                                  device=args.device,
                                  sort_key=lambda x: len(x.message),
                                  sort_within_batch=False,
                                  repeat=False)
    args.dataset_size = len(train_loader.dataset)
    args.dataloader_size = len(train_loader)

    # get some random training elements
    dataiter = train_loader.__iter__()

    # Show sample of messages
    if args.plot:
        batch = next(dataiter)
        print_batch(batch)

    # Define optimizer
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam(args.net.parameters(), lr=.01)
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD(args.net.parameters(), lr=.01)

    # Set loss function
    criterion_irony = torch.nn.BCELoss()
    criterion_topic = torch.nn.NLLLoss()

    # Restore past checkpoint
    restore_checkpoint()

    # Set best for minimization
    args.best = float('inf')

    print('Started Training')
    # loop over the dataset multiple times
    for epoch in range(args.epochs):

        # New epoch
        train_loader.init_epoch()
        dataiter = train_loader.__iter__()

        # reset running loss statistics
        args.train_loss = args.train_acc = args.running_loss = 0.0

        for batch_idx, batch in enumerate(dataiter, 1):

            # Get text data
            inputs = batch.__dict__['message']
            irony = batch.__dict__['is_ironic'].float()
            topic = batch.__dict__['topic']

            # Send to device
            inputs = inputs.to(args.device)
            irony = irony.to(args.device)
            topic = topic.to(args.device)

            # Calculate gradients and update
            with autograd.detect_anomaly():
                # zero the parameter gradients
                args.optimizer.zero_grad()

                # forward + backward + optimize
                (output_irony, output_topic) = args.net(inputs)
                loss_irony = criterion_irony(output_irony, irony)
                loss_topic = criterion_topic(output_topic, topic)
                loss = loss_irony + loss_topic

                loss.backward()
                args.optimizer.step()

            # Log batch status
            batch_status(batch_idx, epoch, train_loader,
                         loss, validationset,
                         (output_irony, irony, loss_irony),
                         (output_topic, topic, loss_topic))

        print('====> Epoch: {} Average loss: {:.4f} Average acc {:.4f}%'
              .format(epoch,
                      args.train_loss / len(train_loader),
                      100. * args.train_acc / len(train_loader)))

    # Add trained model
    print('Finished Training')


def validate(validationset, print_info=False, log_info=False, global_step=0):
    # Create dataset loader
    valid_loader = BucketIterator(validationset,
                                  batch_size=args.batch_size,
                                  device=args.device,
                                  sort_key=lambda x: len(x.message),
                                  sort_within_batch=False,
                                  repeat=False)

    # iterator
    dataiter = valid_loader.__iter__()

    if print_info:
        print('Started Validation')

    correct = total = 0
    for batch_idx, batch in enumerate(dataiter, 1):

        # Get text data
        inputs = batch.__dict__['message']
        irony = batch.__dict__['is_ironic'].float()

        # Send to device
        inputs = inputs.to(args.device)

        # forward + predict
        (output_irony, output_topic) = args.net(inputs)
        predicted = (output_irony > .5)

        # Sum of predicted
        correct += (predicted.type(torch.long) ==
                    irony.type(torch.long)).sum().item()
        total += len(irony)

        if batch_idx == 1 and print_info:
            print_batch(batch, irony.cpu(), predicted.cpu(), args)

    if log_info:
        args.writer.add_scalar('Train/validation_acc',
                               100 * correct / total,
                               global_step)

    if print_info:
        print('Accuracy of the network on %d messages: %d %%' % (
            len(validationset), 100 * correct / total))

        # Add trained model
        print('Finished Validation')


def restore_checkpoint():
    if args.checkpoint == 'none':
        return

    # Load provided checkpoint
    checkpoint = torch.load(args.checkpoint)
    print('Restored weights from {}.'.format(args.checkpoint))

    # Restore past checkpoint
    args.net.load_state_dict(checkpoint['net_state_dict'])
    args.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # To continue training
    args.net.train()


def process_checkpoint(loss, targets, outputs, global_step):

    # check if current batch had best generating fitness
    steps_before_best = 100
    if loss < args.best and global_step > steps_before_best:
        args.best = loss

        # Save best checkpoint
        torch.save({
            'net_state_dict': args.net.state_dict(),
            'optimizer_state_dict': args.optimizer.state_dict(),
        }, "checkpoint/best_{}.pt".format(args.run))

        # Write tensorboard statistics
        args.writer.add_scalar('Best/loss', loss, global_step)

    # Save current checkpoint
    torch.save({
        'net_state_dict': args.net.state_dict(),
        'optimizer_state_dict': args.optimizer.state_dict(),
    }, "checkpoint/last_{}.pt".format(args.run))


def create_run_name():
    run = '{}={}'.format('nw', args.network)
    run += '_{}={}'.format('ds', args.dataset)
    run += '_{}={}'.format('op', args.optimizer)
    run += '_{}={}'.format('ep', args.epochs)
    run += '_{}={}'.format('bs', args.batch_size)
    run += '_{}={}'.format('tp', args.train_percentage)
    run += '_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))

    return run


def main():
    # Printing parameters
    torch.set_printoptions(precision=10)
    torch.set_printoptions(edgeitems=5)

    print(torch.cuda.is_available())

    # Set up GPU
    if args.device != 'cpu':
        args.device = torch.device('cuda:0'
                                   if torch.cuda.is_available()
                                   else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    # Dataset information
    print('device : {}'.format(args.device))

    # Save parameters in string to name the execution
    args.run = create_run_name()

    # print run name
    print('execution name : {}'.format(args.run))

    # Tensorboard summary writer
    args.writer = SummaryWriter('runs/' + args.run)

    # Load dataset
    if args.dataset == 'irosva-mx':
        train_path = 'IroSvA2019/train/irosva.mx.train.csv'
    if args.dataset == 'irosva-es':
        train_path = 'IroSvA2019/train/irosva.es.train.csv'
    if args.dataset == 'irosva-cu':
        train_path = 'IroSvA2019/train/irosva.cu.train.csv'

    trainset = load_dataset(train_path, args)

    # Split dataset
    trn, vld = trainset.split(args.train_percentage)

    # Dataset information
    print('train dataset : {} elements'.format(len(trn)))
    print('validate dataset : {} elements'.format(len(vld)))

    # Create network
    if args.network == 'lstm':
        net = LSTM_MTL_Classifier(args.embedding,
                                  args.hidden,
                                  len(args.TEXT.vocab),
                                  args.layers,
                                  args.dropout,
                                  len(args.TOPIC.vocab))

    # Send networks to device
    args.net = net.to(args.device)

    if args.summary:
        print(args.net)

    # Train network
    train(trn, vld)

    # Vaidate trainning
    validate(vld, print_info=True)

    # Close tensorboard writer
    args.writer.close()


if __name__ == "__main__":
    main()
