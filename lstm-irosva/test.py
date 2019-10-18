# -*- coding: utf-8 -*-
import argparse

import torch
from torchtext.data import *

# Import network
from network import *
from utils import *

# Parser arguments
parser = argparse.ArgumentParser(description='Test PyTorch LSTM '
                                             'Irony Classifier')
parser.add_argument('--checkpoint', '--check',
                    default='none',
                    help='path to checkpoint to be restored for inference')
parser.add_argument('--batch-size', '--b',
                    type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 16)')
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
parser.add_argument('--vocabulary', '--v',
                    type=int, default=5000, metavar='N',
                    help='size of vocabulary (default: 5000)')
parser.add_argument('--dataset', '--data',
                    default='irosva-mx',
                    choices=['irosva-mx', 'irosva-es', 'irosva-cu'],
                    help='pick a specific dataset (default: "irosva-mx")')
parser.add_argument('--summary', '--sm',
                    action='store_true',
                    help='show summary of model')
parser.add_argument('--plot', '--p',
                    action='store_true',
                    help='plot dataset sample')
args = parser.parse_args()
print(args)


def test(testset):
    # Create dataset loader
    test_loader = BucketIterator(testset,
                                 batch_size=args.batch_size,
                                 device=args.device,
                                 sort_key=lambda x: len(x.message),
                                 sort_within_batch=False,
                                 repeat=False,
                                 train=False)

    # iterator
    dataiter = test_loader.__iter__()

    # Restore past checkpoint
    restore_checkpoint()

    print('Started Validation')
    correct = total = 0
    for batch_idx, batch in enumerate(dataiter, 1):

        # Get text data
        inputs = batch.__dict__['message']
        targets = batch.__dict__['is_ironic'].float()

        # Send to device
        inputs = inputs.to(args.device)

        # forward + predict
        outputs = args.net(inputs)
        predicted = (outputs > .5)

        # Sum of predicted
        predicted_cpu = predicted.cpu().type(torch.long)
        targets = targets.cpu().type(torch.long)
        correct += (predicted_cpu == targets).sum().item()
        total += len(targets)

        if batch_idx == 1:
            print_batch(batch, targets, predicted_cpu, args)

    print(correct, total)
    print('Accuracy of the network on the %d test messages: %d %%' % (
        len(testset), 100 * correct / total))

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

    # To do inference
    args.net.eval()


def main():
    # Printing parameters
    torch.set_printoptions(precision=10)
    torch.set_printoptions(edgeitems=5)

    # Set up GPU
    if args.device != 'cpu':
        args.device = torch.device('cuda:0'
                                   if torch.cuda.is_available()
                                   else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print('device : {}'.format(args.device))

    # Load dataset
    if args.dataset == 'irosva-mx':
        test_path = 'IroSvA2019/train/irosva.mx.train.csv'
    if args.dataset == 'irosva-es':
        test_path = 'IroSvA2019/train/irosva.es.train.csv'
    if args.dataset == 'irosva-cu':
        test_path = 'IroSvA2019/train/irosva.cu.train.csv'

    testset = load_dataset(test_path, args)

    # Dataset information
    print('test dataset : {} elements'.format(len(testset)))

    # Create network
    if args.network == 'lstm':
        net = LSTM_Irony_Classifier(args.batch_size,
                                    args.embedding,
                                    args.hidden,
                                    len(args.TEXT.vocab),
                                    args.layers)

    # Send networks to device
    args.net = net.to(args.device)

    if args.summary:
        print(args.net)

    # Test the trained model if provided
    if args.checkpoint != 'none':
        test(testset)


if __name__ == "__main__":
    main()
