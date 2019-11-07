# -*- coding: utf-8 -*-
import warnings
from datetime import datetime
from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning

from torchtext import vocab
from torchtext.data import *
from torchtext.datasets import *

# Import network
from network import *

# Filter scikit-learn metric warnings
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


def load_dataset(train_path, valid_path, test_path, args):

    # Create new fields on vocab
    args.WORD = Field(init_token="<bos>", eos_token="<eos>", lower=True)
    args.TAG = Field(init_token="<bos>", eos_token="<eos>")

    # Data fields
    data_fields = (('word', args.WORD), ('tag', args.TAG))

    # Create datasets
    trn_data = SequenceTaggingDataset(train_path, fields=data_fields)
    vld_data = SequenceTaggingDataset(valid_path, fields=data_fields)
    tst_data = SequenceTaggingDataset(test_path, fields=data_fields)

    # Merge train + test to get full vocab
    list_of_trn = [x for x in trn_data]
    list_of_vld = [x for x in vld_data]
    list_of_tst = [x for x in tst_data]
    list_of_join = list_of_trn + list_of_vld + list_of_tst
    full_dataset = Dataset(list_of_join, data_fields)

    # build vocaboulary
    if args.glove:
        args.WORD.build_vocab(full_dataset,
                              vectors=vocab.GloVe('twitter.27B',
                                                  dim=args.embedding))
    else:
        args.WORD.build_vocab(full_dataset)

    args.TAG.build_vocab(trn_data)

    # Add parameters
    args.lenword = len(args.WORD.vocab)
    args.lentag = len(args.TAG.vocab)

    print('vocabulary length : ', args.lenword)
    print('number of tags : ', args.lentag)

    total = 0
    for key, value in args.TAG.vocab.freqs.items():
        total += value
    for key, value in args.TAG.vocab.freqs.items():
        print('\t{} {} elems {:.2f}%'.format(key, value, 100 * value / total))

    # Dataset information
    print('train dataset : {} elements'.format(len(trn_data)))
    print('validate dataset : {} elements'.format(len(vld_data)))
    print('test dataset : {} elements'.format(len(tst_data)))

    return (trn_data, vld_data, tst_data)


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


def print_batch(inputs, targets, predicted, args):
    # print prediction
    for idx in range(predicted.shape[0]):
        for jdx in range(predicted.shape[1]):
            word = args.WORD.vocab.itos[inputs[idx, jdx]].encode()[2: -1]
            truth = args.TAG.vocab.itos[targets[idx, jdx]].encode()[2: -1]
            pred = args.TAG.vocab.itos[predicted[idx, jdx]].encode()[2: -1]

            if word not in['<pad>', '<bos>', '<eos>']:
                print('{}\t{}\t{}'.format(str(word), str(truth), str(pred)),
                      end='\n')
        print('')


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
           isinstance(value, float):
            hparams[key] = value
    return hparams


def read_checkpoint(args):
    if args.checkpoint == 'none':
        return

    # Load provided checkpoint
    checkpoint = torch.load(args.checkpoint)
    print('Read weights from {}.'.format(args.checkpoint))

    # Discard hparams
    discard = ['run', 'predict', 'checkpoint', 'WORD', 'TAG', 'summary']

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
    run += '_{}={}'.format('hd', args.hidden)
    run += '_{}={}'.format('em', args.embedding)
    run += '_{}={}'.format('ep', args.epochs)
    run += '_{}={}'.format('bs', args.batch_size)
    run += '_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))

    return run
