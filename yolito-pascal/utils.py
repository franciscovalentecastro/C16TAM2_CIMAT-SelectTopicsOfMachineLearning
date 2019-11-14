# -*- coding: utf-8 -*-
import torch
import warnings
from datetime import datetime
from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning

# Import network
from network import *

# Filter scikit-learn metric warnings
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


def split_dataset(dataset, args):
    # Index of data subsets. 1000 test elems
    trnvld_idx = list(range(0, len(dataset) - 1000))
    tst_idx = list(range(len(dataset) - 1000, len(dataset)))

    trn_size = int(args.train_percentage * len(trnvld_idx))
    vld_size = len(trnvld_idx) - trn_size

    # Subset train, valid, test dataset
    trnvld_sub = torch.utils.data.Subset(dataset, trnvld_idx)
    trn, vld = torch.utils.data.random_split(trnvld_sub, [trn_size, vld_size])
    tst = torch.utils.data.Subset(dataset, tst_idx)

    # Dataset information
    print('train dataset : {} elements'.format(len(trn)))
    print('validate dataset : {} elements'.format(len(vld)))
    print('test dataset : {} elements'.format(len(tst)))

    return trn, vld, tst


def get_hparams(dictionary):
    hparams = {}
    for key, value in dictionary.items():
        if isinstance(value, int) or \
           isinstance(value, str) or \
           isinstance(value, float) or \
           isinstance(value, list):
            hparams[key] = value
    return hparams


def read_checkpoint(args):
    if args.checkpoint == 'none':
        return

    # Load provided checkpoint
    checkpoint = torch.load(args.checkpoint)
    print('Read weights from {}.'.format(args.checkpoint))

    # Discard hparams
    discard = ['run', 'predict', 'checkpoint', 'summary']

    # Restore past checkpoint
    hparams = checkpoint['hparams']
    for key, value in hparams.items():
        if (key not in discard):
            args.__dict__[key] = value


def restore_checkpoint(args):
    if args.checkpoint == 'none':
        return

    # Load provided checkpoint
    checkpoint = torch.load(args.checkpoint)
    print('Restored weights from {}.'.format(args.checkpoint))

    # Restore weights
    args.net.load_state_dict(checkpoint['net_state_dict'])

    if args.predict:
        # To do inference
        args.net.eval()
    else:
        # Read optimizer parameters
        args.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # To continue training
        args.net.train()


def process_checkpoint(loss, global_step, args):
    # check if current batch had best generating fitness
    steps_before_best = 100
    if loss < args.best and global_step > steps_before_best:
        args.best = loss

        # Save best checkpoint
        torch.save({
            'net_state_dict': args.net.state_dict(),
            'optimizer_state_dict': args.optimizer.state_dict(),
            'hparams': args.hparams,
        }, "checkpoint/best_{}.pt".format(args.run))

        # Write tensorboard statistics
        args.writer.add_scalar('Best/loss', loss, global_step)

    # Save current checkpoint
    torch.save({
        'net_state_dict': args.net.state_dict(),
        'optimizer_state_dict': args.optimizer.state_dict(),
        'hparams': args.hparams,
    }, "checkpoint/last_{}.pt".format(args.run))


def create_run_name(args):
    run = '{}={}'.format('nw', args.network)
    run += '_{}={}'.format('ds', args.dataset)
    run += '_{}={}'.format('op', args.optimizer)
    run += '_{}={}'.format('ep', args.epochs)
    run += '_{}={}'.format('bs', args.batch_size)
    run += '_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))

    return run
