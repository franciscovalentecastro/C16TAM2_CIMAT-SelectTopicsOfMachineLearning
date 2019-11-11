# -*- coding: utf-8 -*-
import warnings
from datetime import datetime
from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning

# Import network
from network import *

# Filter scikit-learn metric warnings
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


def write_batch(inputs, predicted, file, tst, args):
    # print prediction
    for idx in range(predicted.shape[0]):
        for jdx in range(predicted.shape[1]):
            word = args.WORD.vocab.itos[inputs[idx, jdx]]
            pred = args.TAG.vocab.itos[predicted[idx, jdx]]

            if word not in['<pad>', '<bos>', '<eos>']:
                line = tst.readline()[:-1]
                file.write('{}\t{}\n'.format(line, pred))

        line = tst.readline()
        print('', file=file)


def predict(outputs, targets):
    if type(outputs) is tuple:
        ne, is_ne = outputs
        targets, _ = targets

        # predict by getting max from dist
        predicted = (ne.argmax(dim=1)).type(torch.long)

        # Use other task to predict
        predicted[is_ne < 0.5] = 4

    else:
        # predict by getting max from dist
        predicted = (outputs.argmax(dim=1)).type(torch.long)

    # Change unwanted tag predictions
    for idx in range(4):
        predicted[predicted == idx] = 4

    return predicted, targets


def calculate_metrics(targets, predictions, args, report=True):
    # Calculate metrics
    avg = 'macro'

    # remove unwanted tags
    for idx in range(4):
        dif = targets != idx
        targets = targets[dif]
        predictions = predictions[dif]

    # ignore scikit-learn metrics warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # calculate metrics
        met = {
            'acc': metrics.accuracy_score(targets, predictions),
            'bacc': metrics.balanced_accuracy_score(targets, predictions),
            'prec': metrics.precision_score(targets, predictions, average=avg),
            'rec': metrics.recall_score(targets, predictions, average=avg),
            'f1': metrics.f1_score(targets, predictions, average=avg)}

    # Classification report
    if report:
        # Labels to predict and names
        labels = range(4, 17)
        names = args.TAG.vocab.itos[4:]

        # Print classification report
        print(metrics.classification_report(targets, predictions,
                                            labels, names))
    return met


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
