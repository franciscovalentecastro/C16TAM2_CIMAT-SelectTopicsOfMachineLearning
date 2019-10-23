# -*- coding: utf-8 -*-
from datetime import datetime

from torchtext import vocab
from torchtext.data import *

# Import network
from network import *


def load_dataset(path, args, train=True, build_vocab=True):

    def tokenize(x):
        # Characters to exclude
        exclude = u'"%\'()*+,-./:;<=>[\]^_`{|}~'

        # Remove punctuation
        x = x.translate(str.maketrans('', '', exclude))

        return x.split()

    # Create new fields on vocab
    if build_vocab:
        args.TEXT = Field(sequential=True,
                          tokenize=tokenize,
                          lower=True)
        args.TOPIC = Field(sequential=False,
                           use_vocab=True,
                           lower=True)
        args.LABEL = Field(sequential=False,
                           use_vocab=False)

    if args.dataset == 'haha':
        datafields = [('id', None),
                      ('text', args.TEXT),
                      ('is_humor', args.LABEL),
                      ('votes_no', None),
                      ('votes_1', None),
                      ('votes_2', None),
                      ('votes_3', None),
                      ('votes_4', None),
                      ('votes_5', None),
                      ('funniness_average', None)]
    else:
        datafields = [('id', None),
                      ('topic', args.TOPIC),
                      ('is_ironic', args.LABEL),
                      ('message', args.TEXT)]

    dataset = TabularDataset(path=path,
                             format='CSV',
                             skip_header=True,
                             fields=datafields)

    # build vocaboulary
    if build_vocab:
        if args.fasttext:
            args.TEXT.build_vocab(dataset, min_freq=2,
                                  vectors=vocab.FastText('es'))
        else:
            args.TEXT.build_vocab(dataset, min_freq=2)

        args.TOPIC.build_vocab(dataset)

        print('vocabulary length : ', len(args.TEXT.vocab))
        print('number of topics : ', len(args.TOPIC.vocab))

    if train:
        # Split dataset
        trn, vld = dataset.split(args.train_percentage)

        target = 'is_humor' if args.dataset == 'haha' else 'is_ironic'

        # study dataset
        vld_num_target = trn_num_target = 0
        for element in trn:
            trn_num_target += int(element.__dict__[target])
        trn_num_not_target = len(trn) - trn_num_target
        for element in vld:
            vld_num_target += int(element.__dict__[target])
        vld_num_not_target = len(vld) - vld_num_target

        # Dataset information
        print('train dataset : {} elements'.format(len(trn)))
        print('train dataset ({}): {} elements. {:.2f}%'
              .format(target, trn_num_target,
                      100 * trn_num_target / len(trn)))
        print('train dataset (not {}): {} elements. {:.2f}%'
              .format(target, trn_num_not_target,
                      100 * trn_num_not_target / len(trn)))
        print('validate dataset : {} elements'.format(len(vld)))
        print('validate dataset ({}): {} elements. {:.2f}%'
              .format(target, vld_num_target,
                      100 * vld_num_target / len(vld)))
        print('validate dataset (not {}): {} elements. {:.2f}%'
              .format(target, vld_num_not_target,
                      100 * vld_num_not_target / len(vld)))

        return (trn, vld)

    else:  # Return test dataset
        tst = dataset
        target = 'is_humor' if args.dataset == 'haha' else 'is_ironic'

        # study dataset
        tst_num_target = 0
        for element in tst:
            tst_num_target += int(element.__dict__[target])
        tst_num_not_target = len(tst) - tst_num_target

        # Dataset information
        print('test dataset : {} elements'.format(len(tst)))
        print('test dataset ({}): {} elements. {:.2f}%'
              .format(target, tst_num_target,
                      100 * tst_num_target / len(tst)))
        print('test dataset (not {}): {} elements. {:.2f}%'
              .format(target, tst_num_not_target,
                      100 * tst_num_not_target / len(tst)))

        return tst


def print_batch(batch, targets, predicted, args):
    if args.dataset == 'haha':
        text = batch.__dict__['text']
    else:
        text = batch.__dict__['message']
        topic = batch.__dict__['topic']

    for idx, t in enumerate(torch.t(text)):
        if args.dataset == 'haha':
            # Irony
            print('IS_HUMOR : ', int(targets[idx]), end=', ')
            print('PREDICTED : ', int(predicted[idx]), end=', ')
        else:
            # Topic
            topic_word = str(args.TOPIC.vocab.itos[int(topic[idx])])
            print('TOPIC : ({}){}'
                  .format(int(topic[idx]), topic_word), end=', ')

            # Irony
            print('IS_IRONIC : ', int(targets[idx]), end=', ')
            print('PREDICTED : ', int(predicted[idx]), end=', ')

        # Message
        print(' MESSAGE : ', end='')
        for elem in t:
            word = str(args.TEXT.vocab.itos[int(elem)])
            if word != '<pad>':
                print('{}'.format(str(word)), end=' ')
        print('\n\n')


def read_checkpoint(args):
    if args.checkpoint == 'none':
        return

    # Load provided checkpoint
    checkpoint = torch.load(args.checkpoint)
    print('Read weights from {}.'.format(args.checkpoint))

    # Restore past checkpoint
    args.hparams = checkpoint['hparams']
    for key, value in args.hparams.items():
        if (isinstance(key, int) or isinstance(key, str) or
            isinstance(key, float)) and key != 'run' and \
                key != 'test' and key != 'checkpoint' and \
                key != 'TEXT' and key != 'TOPIC' and \
                key != 'LABEL':
            args.__dict__[key] = value


def restore_checkpoint(args):
    if args.checkpoint == 'none':
        return

    # Load provided checkpoint
    checkpoint = torch.load(args.checkpoint)
    print('Restored weights from {}.'.format(args.checkpoint))

    # Restore weights
    args.net.load_state_dict(checkpoint['net_state_dict'])
    args.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # To continue training
    args.net.train()


def restore_checkpoint_eval(args):
    if args.checkpoint == 'none':
        return

    # Load provided checkpoint
    checkpoint = torch.load(args.checkpoint)
    print('Restored weights from {}.'.format(args.checkpoint))

    # Restore weights
    args.net.load_state_dict(checkpoint['net_state_dict'])

    # To do inference
    args.net.eval()


def process_checkpoint(loss, targets, outputs, global_step, args):
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
    run += '_{}={}'.format('tp', args.train_percentage)
    run += '_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))

    return run
