# -*- coding: utf-8 -*-
import argparse
from sklearn import metrics

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
                    default='mtl',
                    choices=['mtl', 'lstm'],
                    help='pick a specific network to train (default: mtl)')
parser.add_argument('--embedding', '--emb',
                    type=int, default=100, metavar='N',
                    help='dimension of embedding (default: 100)')
parser.add_argument('--hidden', '--h',
                    type=int, default=10, metavar='N',
                    help='dimension of hidden state (default: 10)')
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
                    default='irosva-mx',
                    choices=['irosva-mx', 'irosva-es',
                             'irosva-cu', 'irosva-comb', 'haha'],
                    help='pick a specific dataset (default: "irosva-mx")')
parser.add_argument('--fasttext', '--ft',
                    action='store_true',
                    help='use pretrained embeddings')
parser.add_argument('--checkpoint', '--check',
                    default='none',
                    help='path to checkpoint to be restored')
parser.add_argument('--test', '--tst',
                    action='store_true',
                    help='test model')
parser.add_argument('--plot', '--p',
                    action='store_true',
                    help='plot dataset sample')
parser.add_argument('--summary', '--sm',
                    action='store_true',
                    help='show summary of model')
args = parser.parse_args()


def predict(outputs, targets):
    targets = targets.type(torch.long)
    predicted = (outputs > .5).type(torch.long)
    correct = (predicted == targets).sum().item()
    args.train_acc += correct / args.batch_size

    return (predicted, correct)


def batch_status(batch_idx, epoch, train_loader, loss,
                 validationset, irony, topic, humor):
    # Unpack information
    out_irony, trgt_irony, loss_irony = irony
    out_topic, trgt_topic, loss_topic = topic
    out_humor, trgt_humor, loss_humor = humor

    # Global step
    global_step = batch_idx + len(train_loader) * epoch

    # update running loss statistics
    args.running_loss += loss.item()
    args.train_loss += loss.item()

    # predict
    if args.dataset == 'haha':
        (predicted, correct) = predict(out_humor, trgt_humor)
    else:
        (predicted, correct) = predict(out_irony, trgt_irony)

    # Write tensorboard statistics
    args.writer.add_scalar('Train/loss_target', loss.item(), global_step)
    args.writer.add_scalar('Train/loss_irony', loss_irony, global_step)
    args.writer.add_scalar('Train/loss_topic', loss_topic, global_step)
    args.writer.add_scalar('Train/loss_humor', loss_humor, global_step)

    # print every args.log_interval of batches
    if global_step % args.log_interval == 0:
        # validate
        acc = validate(validationset, log_info=True, global_step=global_step)

        # Process current checkpoint
        if args.dataset == 'haha':
            process_checkpoint(loss.item(), trgt_humor, out_humor,
                               global_step, args)
        else:
            process_checkpoint(loss.item(), trgt_irony, out_irony,
                               global_step, args)

        print('Epoch : {} Batch : {} [{}/{} ({:.0f}%)]\n'
              '====> Run_Loss : {:.4f} Batch_Acc : {:.2f}% '
              'Valid_Acc : {:.2f}%\n'
              '====> Loss : {:.4f} Loss_Irony : {:.4f} '
              'Loss_Topic : {:.4f} Loss_Humor : {:.4f}'
              .format(epoch, batch_idx,
                      args.batch_size * batch_idx,
                      args.dataset_size,
                      100. * batch_idx / args.dataloader_size,
                      args.running_loss / args.log_interval,
                      100. * correct / args.batch_size, 100 * acc,
                      loss, loss_irony, loss_topic, loss_humor),
              end='\n\n')

        args.running_loss = 0.0

    # Pass all pending items to disk
    # args.writer.flush()


def unpack_batch(batch):
    inputs = irony = topic = humor = None
    if args.dataset == 'haha':
        # Get text data
        inputs = batch.__dict__['text']
        humor = batch.__dict__['is_humor'].float()

        # Send to device
        inputs = inputs.to(args.device)
        humor = humor.to(args.device)

    else:
        # Get text data
        inputs = batch.__dict__['message']
        irony = batch.__dict__['is_ironic'].float()
        topic = batch.__dict__['topic']

        # Send to device
        inputs = inputs.to(args.device)
        irony = irony.to(args.device)
        topic = topic.to(args.device)

    return (inputs, irony, topic, humor)


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
        args.optimizer = optim.Adam(args.net.parameters(),
                                    lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD(args.net.parameters(),
                                   lr=args.learning_rate)

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
            inputs, irony, topic, humor = unpack_batch(batch)

            # Calculate gradients and update
            with autograd.detect_anomaly():
                # zero the parameter gradients
                args.optimizer.zero_grad()

                # forward + backward + optimize
                out_irony = out_topic = out_humor = None
                loss_irony_val = loss_topic_val = loss_humor_val = 0
                if args.network == 'lstm':
                    out_irony = args.net(inputs)

                    # calculate loss
                    loss_irony = args.criterion_irony(out_irony, irony)
                    loss = loss_irony

                    loss_irony_val = loss_irony.item()

                elif args.network == 'mtl':
                    (out_irony, out_topic, out_humor) = args.net(inputs)

                    # calculate loss
                    if args.dataset == 'haha':
                        loss_humor = args.criterion_humor(out_humor, humor)
                        loss = loss_humor

                        loss_humor_val = loss_humor.item()
                    else:
                        loss_irony = args.criterion_irony(out_irony, irony)
                        loss_topic = args.criterion_topic(out_topic, topic)
                        loss = loss_irony + loss_topic

                        loss_irony_val = loss_irony.item()
                        loss_topic_val = loss_topic.item()

                loss.backward()
                args.optimizer.step()

            # Pack batch results
            irony_result = (out_irony, irony, loss_irony_val)
            topic_result = (out_topic, topic, loss_topic_val)
            humor_result = (out_humor, humor, loss_humor_val)

            # Log batch status
            batch_status(batch_idx, epoch, train_loader,
                         loss, validationset,
                         irony_result, topic_result, humor_result)

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

    loss = 0
    labels = torch.tensor([0], dtype=torch.int)
    predictions = torch.tensor([0], dtype=torch.int)
    for batch_idx, batch in enumerate(dataiter, 1):
        # Unpack batch
        inputs, irony, topic, humor = unpack_batch(batch)

        # forward + backward + optimize
        out_irony = out_topic = out_humor = None
        if args.network == 'lstm':
            out_irony = args.net(inputs)
        elif args.network == 'mtl':
            (out_irony, out_topic, out_humor) = args.net(inputs)

        # calculate loss
        if args.dataset == 'haha':
            targets = humor
            predicted = (out_humor > .5)
            loss += args.criterion_humor(out_humor, humor)
        else:
            targets = irony
            predicted = (out_irony > .5)
            loss += args.criterion_irony(out_irony, irony)

        # Concatenate prediction and truth
        predictions = torch.cat((predictions, predicted.int().cpu()))
        labels = torch.cat((labels, targets.int().cpu()))

        if batch_idx == 1 and print_info:
            print_batch(batch, targets.cpu(), predicted.cpu(), args)

    # Calculate metrics
    acc = metrics.accuracy_score(labels, predictions)
    bacc = metrics.balanced_accuracy_score(labels, predictions)
    prec = metrics.precision_score(labels, predictions)
    rec = metrics.recall_score(labels, predictions)
    f1 = metrics.f1_score(labels, predictions)

    # Classification report
    print(metrics.classification_report(labels, predictions))

    if log_info:
        args.writer.add_scalar('Validation/accuracy',
                               acc, global_step)
        args.writer.add_scalar('Validation/balanced_accuracy',
                               bacc, global_step)
        args.writer.add_scalar('Validation/precision',
                               prec, global_step)
        args.writer.add_scalar('Validation/recall',
                               rec, global_step)
        args.writer.add_scalar('Validation/f1',
                               f1, global_step)
        args.writer.add_scalar('Validation/loss',
                               loss / len(validationset),
                               global_step)

    if print_info:
        print('Accuracy of the network on %d validation messages: %d %%' % (
            len(validationset), 100 * acc))

        # Add trained model
        print('Finished Validation')

    return acc


def test(testset):
    # Create dataset loader
    test_loader = BucketIterator(testset,
                                 batch_size=args.batch_size,
                                 device=args.device,
                                 sort_key=lambda x: len(x.message),
                                 sort_within_batch=False,
                                 repeat=False)

    # iterator
    dataiter = test_loader.__iter__()

    # restore checkpoint
    restore_checkpoint_eval(args)

    print('Started Testing')
    loss = 0
    labels = torch.tensor([0], dtype=torch.int)
    predictions = torch.tensor([0], dtype=torch.int)
    for batch_idx, batch in enumerate(dataiter, 1):
        # Unpack batch
        inputs, irony, topic, humor = unpack_batch(batch)

        # forward + backward + optimize
        out_irony = out_topic = out_humor = None
        if args.network == 'lstm':
            out_irony = args.net(inputs)
        elif args.network == 'mtl':
            (out_irony, out_topic, out_humor) = args.net(inputs)

        # calculate loss
        if args.dataset == 'haha':
            targets = humor
            predicted = (out_humor > .5)
            loss += args.criterion_humor(out_humor, humor)
        else:
            targets = irony
            predicted = (out_irony > .5)
            loss += args.criterion_irony(out_irony, irony)

        # Concatenate prediction and truth
        predictions = torch.cat((predictions, predicted.int().cpu()))
        labels = torch.cat((labels, targets.int().cpu()))

        if batch_idx == 1:
            print_batch(batch, targets.cpu(), predicted.cpu(), args)

    # Calculate metrics
    met = {'acc': metrics.accuracy_score(labels, predictions),
           'bacc': metrics.balanced_accuracy_score(labels, predictions),
           'prec': metrics.precision_score(labels, predictions),
           'rec': metrics.recall_score(labels, predictions),
           'f1': metrics.f1_score(labels, predictions)}

    # Classification report
    print(metrics.classification_report(labels, predictions))

    print('Accuracy of the network on %d test messages: %d %%' % (
        len(testset), 100 * met['acc']))

    for key, value in met.items():
        print('test metric "{}" : {:.2f}'.format(key, value))

    # Add trained model
    print('Finished Testing')


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

    # add hyperparameters to tensorboard
    # hparams = args.__dict__.copy()
    # args.hparams = hparams
    # del hparams['device']
    # writer.add_hparams(hparams, {'metrics': 0})

    # Save as parameter
    args.writer = writer

    # Load dataset
    if args.dataset == 'irosva-mx':
        train_path = 'IroSvA2019/train/irosva.mx.training.csv'
        test_path = 'IroSvA2019/test/irosva.mx.test-w-truth.csv'
    elif args.dataset == 'irosva-es':
        train_path = 'IroSvA2019/train/irosva.es.training.csv'
        test_path = 'IroSvA2019/test/irosva.es.test-w-truth.csv'
    elif args.dataset == 'irosva-cu':
        train_path = 'IroSvA2019/train/irosva.cu.training.csv'
        test_path = 'IroSvA2019/test/irosva.cu.test-w-truth.csv'
    elif args.dataset == 'irosva-comb':
        train_path = 'IroSvA2019/train/irosva.comb.train.csv'
    elif args.dataset == 'haha':
        train_path = 'HAHA2019/haha_2019_train.csv'

    # Read parameters from checkpoint
    if args.checkpoint:
        read_checkpoint(args)

    trn, vld = load_dataset(train_path, args)

    if args.dataset in ['irosva-es', 'irosva-cu', 'irosva-mx']:
        tst = load_dataset(test_path, args, build_vocab=False, train=False)

    # Set fasttext embeddings parameters
    if args.fasttext:
        args.embedding = 300

    # Add parameters
    args.vocab = len(args.TEXT.vocab)
    args.categories = len(args.TOPIC.vocab)

    # Print parameters
    args.hparams = {}
    print("Parameters : ", end='')
    for key, value in args.__dict__.items():
        if isinstance(value, int) or \
           isinstance(value, str) or \
           isinstance(value, float):
            args.hparams[key] = value
            print('{} : {},'.format(key, value), end=' ')
    print()

    # Set loss function
    args.criterion_irony = torch.nn.BCELoss()
    args.criterion_topic = torch.nn.NLLLoss()
    args.criterion_humor = torch.nn.BCELoss()

    # Create network
    if args.network == 'mtl':
        net = LSTM_MTL_Classifier(args.embedding,
                                  args.hidden,
                                  args.vocab,
                                  args.layers,
                                  args.dropout,
                                  args.categories)
    if args.network == 'lstm':
        net = LSTM_Irony_Classifier(args.embedding,
                                    args.hidden,
                                    args.vocab,
                                    args.layers,
                                    args.dropout)

    # Load pretrained embeddings
    if args.fasttext:
        net.embedding.weight.data.copy_(args.TEXT.vocab.vectors)
        # net.embedding.weight.requires_grad = False

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

    if args.test:
        # Test network
        test(tst)

    else:
        # Train network
        train(trn, vld)

        # Vaidate trainning
        validate(vld, print_info=True)

    # Delete model + Free memory
    del args.net
    torch.cuda.empty_cache()

    # Close tensorboard writer
    args.writer.close()


if __name__ == "__main__":
    main()
