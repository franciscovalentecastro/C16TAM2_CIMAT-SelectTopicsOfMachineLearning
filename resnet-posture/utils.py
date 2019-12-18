# -*- coding: utf-8 -*-
import torch
import warnings
from datetime import datetime
from sklearn.exceptions import UndefinedMetricWarning

# Import network
from network import *
from datasets import *

# Filter scikit-learn metric warnings
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


def get_max(x, dim=(2, 3)):
    b = x.shape[0]
    j = x.shape[1]
    d = tensor.shape[2]
    m = x.view(b, j, -1).argmax(1)
    indices = torch.cat(((m // d).view(-1, 1),
                         (m % d).view(-1, 1)),
                        dim=1)
    return indices


def load_dataset(args):
    # Initial parameters
    dataDir = 'coco'

    # Load train
    dataType = 'train2017'
    trn = CocoKeypoints('{}/images/{}/'.format(dataDir, dataType,),
                        '{}/annotations/person_keypoints_{}.json'
                        .format(dataDir, dataType), args)
    train_size = int(args.train_percentage * len(trn))
    test_size = len(trn) - train_size
    trn, _ = torch.utils.data.random_split(trn, [train_size, test_size])

    # Load validation
    dataType = 'val2017'
    print('{}/annotations/person_keypoints_{}.json'
          .format(dataDir, dataType))
    vld = CocoKeypoints('{}/images/{}/'.format(dataDir, dataType,),
                        '{}/annotations/person_keypoints_{}.json'
                        .format(dataDir, dataType), args)
    vld = torch.utils.data.Subset(vld, list(range(0, 100)))

    # Load test
    tst = vld

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
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
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
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
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
