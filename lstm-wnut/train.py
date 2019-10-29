# -*- coding: utf-8 -*-
import argparse
from pprint import pprint

import torch
import torch.optim as optim
import torch.autograd as autograd

from torchtext.data import *
from torch.utils.tensorboard import SummaryWriter

# Import network
from network import *
from utils import *

# Parser arguments
parser = argparse.ArgumentParser(description='Train PyTorch LSTM WNut2017 '
                                             'Named Entity Recognition')
parser.add_argument('--batch-size', '--b',
                    type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--log-interval', '--li',
                    type=int, default=50, metavar='N',
                    help='how many batches to wait' +
                         'before logging training status')
parser.add_argument('--epochs', '--e',
                    type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--device', '--d',
                    default='cpu', choices=['cpu', 'cuda'],
                    help='pick device to run the training (defalut: "cpu")')
parser.add_argument('--network', '--n',
                    default='lstm',
                    choices=['lstm', 'teacher', 'mtl', 'attention', 'prop'],
                    help='pick a specific network to train (default: lstm)')
parser.add_argument('--embedding', '--emb',
                    type=int, default=100, metavar='N',
                    help='dimension of embedding (default: 100)')
parser.add_argument('--hidden', '--h',
                    type=int, default=50, metavar='N',
                    help='dimension of hidden state (default: 50)')
parser.add_argument('--layers', '--ly',
                    type=int, default=2, metavar='N',
                    help='dimension of embedding (default: 2)')
parser.add_argument('--dropout', '--drop',
                    type=float, default=.25, metavar='N',
                    help='dropout percentage in lstm layers (default: .25)')
parser.add_argument('--optimizer', '--o',
                    default='adam', choices=['adam', 'sgd'],
                    help='pick a specific optimizer (default: "adam")')
parser.add_argument('--learning-rate', '--lr',
                    type=float, default=.0001, metavar='N',
                    help='learning rate of model (default: .0001)')
parser.add_argument('--dataset', '--data',
                    default='wnut',
                    choices=['wnut'],
                    help='pick a specific dataset (default: "wnut")')
parser.add_argument('--glove',
                    action='store_true',
                    help='use pretrained embeddings')
parser.add_argument('--checkpoint', '--check',
                    default='none',
                    help='path to checkpoint to be restored')
parser.add_argument('--predict', '--pred',
                    action='store_true',
                    help='predict test dataset')
parser.add_argument('--print', '--p',
                    action='store_true',
                    help='print dataset sample')
parser.add_argument('--summary', '--sm',
                    action='store_true',
                    help='show summary of model')
args = parser.parse_args()


def batch_status(batch_idx, outputs, targets, epoch,
                 train_loader, loss, validationset):
    # Global step
    global_step = batch_idx + len(train_loader) * epoch

    # update running loss statistics
    args.running_loss += loss.item()
    args.train_loss += loss.item()

    # predict
    predicted, targets = predict(outputs, targets)

    # Reshape vectors
    targets = targets.reshape(-1)
    predicted = predicted.reshape(-1)

    # Calculate metrics
    batch_met = calculate_metrics(targets.cpu(), predicted.cpu(), False)
    args.train_acc = batch_met['acc']

    # Write tensorboard statistics
    args.writer.add_scalar('Train/loss', loss.item(), global_step)

    # print every args.log_interval of batches
    if global_step % args.log_interval == 0:
        # validate
        vacc, vloss = validate(validationset, log_info=True,
                               global_step=global_step)

        # Process current checkpoint
        process_checkpoint(loss.item(), global_step, args)

        print('Epoch : {} Batch : {} [{}/{} ({:.0f}%)]\n'
              '====> Run_Loss : {:.4f} Acc_Batch : {:.2f}% '
              'Acc_Valid : {:.2f}%\n'
              '====> Loss_Batch : {:.4f} Loss_Valid : {:.4f} '
              .format(epoch, batch_idx,
                      args.batch_size * batch_idx,
                      args.dataset_size,
                      100. * batch_idx / args.dataloader_size,
                      args.running_loss / args.log_interval,
                      batch_met['acc'], vacc, loss.item(), vloss),
              end='\n\n')

        args.running_loss = 0.0

    # (compatibility issues) Pass all pending items to disk
    # args.writer.flush()


def unpack_batch(batch):
    inputs = targets = None

    # Get text data
    inputs = batch.__dict__['word']
    targets = torch.t(batch.__dict__['tag'])

    # Reshape tensor
    inputs = torch.transpose(inputs, 0, 1)

    # Send to device
    inputs = inputs.to(args.device)
    targets = targets.to(args.device)

    # if multi-task generate is_named_entity
    if args.network == 'mtl':
        targets = (targets, targets > 4)

    return (inputs, targets)


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
    if args.print:
        inpts, trgts = unpack_batch(next(dataiter))
        print_batch(inpts, trgts, trgts, args)

    # Define optimizer
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam(args.net.parameters(),
                                    lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD(args.net.parameters(),
                                   lr=args.learning_rate)

    # Set loss function
    args.criterion = loss_ner(args).loss

    # restore checkpoint
    restore_checkpoint(args)

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
            # Unpack batch
            inputs, targets = unpack_batch(batch)

            # Calculate gradients and update
            with autograd.detect_anomaly():
                # zero the parameter gradients
                args.optimizer.zero_grad()

                # forward
                if args.network == 'teacher':
                    # pass targets to do 'Teacher Forcing'
                    outputs = args.net(inputs, True, targets)
                else:
                    outputs = args.net(inputs)

                # calculate loss
                loss = args.criterion(outputs, targets)

                # backward + step
                loss.backward()
                args.optimizer.step()

            # Log batch status
            batch_status(batch_idx, outputs, targets, epoch,
                         train_loader, loss, validationset)

        print('Epoch: {} Average loss: {:.4f} Average acc {:.4f}%'
              .format(epoch, args.train_loss / len(train_loader),
                      100. * args.train_acc))

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

    run_loss = 0
    trgts = torch.tensor([0], dtype=torch.int)
    preds = torch.tensor([0], dtype=torch.int)
    for batch_idx, batch in enumerate(dataiter, 1):
        # Unpack batch
        inputs, targets = unpack_batch(batch)

        # forward
        outputs = args.net(inputs)

        # calculate loss
        run_loss += args.criterion(outputs, targets).item()

        # predict
        predicted, targets = predict(outputs, targets)

        # concatenate prediction and truth
        preds = torch.cat((preds, predicted.reshape(-1).int().cpu()))
        trgts = torch.cat((trgts, targets.reshape(-1).int().cpu()))

        if batch_idx == 1 and print_info:
            print_batch(inputs, targets, predicted, args)

    # Calculate metrics
    met = calculate_metrics(trgts, preds)

    if log_info:
        args.writer.add_scalar('Validation/accuracy',
                               met['acc'], global_step)
        args.writer.add_scalar('Validation/balanced_accuracy',
                               met['bacc'], global_step)
        args.writer.add_scalar('Validation/precision',
                               met['prec'], global_step)
        args.writer.add_scalar('Validation/recall',
                               met['rec'], global_step)
        args.writer.add_scalar('Validation/f1',
                               met['f1'], global_step)
        args.writer.add_scalar('Validation/loss',
                               run_loss / len(valid_loader),
                               global_step)

    if print_info:
        print('Accuracy of the network on %d validation messages: %d %%' % (
            len(validationset), 100 * met['acc']))

        # Add trained model
        print('Finished Validation')

    return met['acc'], run_loss / len(valid_loader)


def predict_test(testset):
    # Create dataset loader
    test_loader = BucketIterator(testset,
                                 batch_size=args.batch_size,
                                 device=args.device,
                                 train=False,
                                 sort=False)

    # iterator
    dataiter = test_loader.__iter__()

    # restore checkpoint
    restore_checkpoint(args)

    # Output file
    f = open('predictions/pred_{}.pt'.format(args.run), 'w', encoding='utf8')

    print('Generating Test Predictions')
    for batch_idx, batch in enumerate(dataiter, 1):
        # Unpack batch
        inputs, targets = unpack_batch(batch)

        # forward
        outputs = args.net(inputs)

        # predict
        predicted, targets = predict(outputs, targets)

        # print batch
        write_batch(inputs, targets, predicted, f, args)

    f.close()
    print('Finished')


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
    # Dataset information
    print('device : {}'.format(args.device))

    # Save parameters in string to name the execution
    args.run = create_run_name(args)

    # print run name
    print('execution name : {}'.format(args.run))

    # Tensorboard summary writer
    writer = SummaryWriter('runs/' + args.run)

    # Save as parameter
    args.writer = writer

    # Load dataset
    if args.dataset == 'wnut':
        train_path = 'emerging_entities_17/wnut17train.conll'
        valid_path = 'emerging_entities_17/emerging.dev.conll'
        test_path = 'emerging_entities_17/emerging.test.annotated'

    # Read parameters from checkpoint
    if args.checkpoint:
        read_checkpoint(args)

    # Read dataset
    trn, vld, tst = load_dataset(train_path, valid_path, test_path, args)

    # Set glove embeddings parameters
    if args.glove:
        args.embedding = 100

    # Get hparams from args
    args.hparams = get_hparams(args.__dict__)
    print('\nParameters :')
    pprint(args.hparams)
    print()

    # Create network
    if args.network == 'lstm':
        net = LSTM_NER(args.embedding,
                       args.hidden,
                       args.lenword,
                       args.layers,
                       args.dropout,
                       args.lentag)
    elif args.network == 'teacher':
        net = LSTM_NER_TF(args.embedding,
                          args.hidden,
                          args.lenword,
                          args.layers,
                          args.dropout,
                          args.lentag,
                          args.device)
    elif args.network == 'mtl':
        net = LSTM_NER_MTL(args.embedding,
                           args.hidden,
                           args.lenword,
                           args.layers,
                           args.dropout,
                           args.lentag)
    elif args.network == 'prop':
        net = LSTM_NER_Proposal(args.embedding,
                                args.hidden,
                                args.lenword,
                                args.layers,
                                args.dropout,
                                args.lentag)

    # Load pretrained embeddings
    if args.glove:
        net.embedding.weight.data.copy_(args.WORD.vocab.vectors)
        net.embedding.weight.requires_grad = False

    # Send networks to device
    args.net = net.to(args.device)

    # number of parameters
    total_params = sum(p.numel()
                       for p in args.net.parameters() if p.requires_grad)
    print('number of trainable parameters : ', total_params)

    # summarize model layers
    if args.summary:
        print(args.net)
        return

    if args.predict:
        # Predict test
        predict_test(tst)
    else:
        # Train network
        train(trn, vld)

        # Validate trainning
        validate(vld, print_info=args.print)

    # (compatibility issues) Add hparams with metrics to tensorboard
    # args.writer.add_hparams(args.hparams, {'metrics': 0})

    # Delete model + Free memory
    del args.net
    torch.cuda.empty_cache()

    # Close tensorboard writer
    args.writer.close()


if __name__ == "__main__":
    main()
