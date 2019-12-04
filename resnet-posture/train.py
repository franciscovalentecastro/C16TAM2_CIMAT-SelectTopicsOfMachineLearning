# -*- coding: utf-8 -*-
import argparse
from pprint import pprint

import torch
import torch.optim as optim
import torch.autograd as autograd

from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

# Import network
from network import *
from utils import *
from imshow import *
from heatmap import heatmap

# Parser arguments
parser = argparse.ArgumentParser(description='Train Resnet Posture estimation')
parser.add_argument('--train-percentage', '--t',
                    type=float, default=.9, metavar='N',
                    help='porcentage of the training set to use (default: .9)')
parser.add_argument('--batch-size', '--b',
                    type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
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
                    default='posture',
                    choices=['posture'],
                    help='pick a specific network to train'
                         '(default: "posture")')
parser.add_argument('--image-shape', '--imshape',
                    type=int, nargs='+',
                    default=[256, 192],
                    metavar='height width',
                    help='rectanlge size to crop input images '
                         '(default: 256 192)')
parser.add_argument('--optimizer', '--o',
                    default='adam', choices=['adam', 'sgd'],
                    help='pick a specific optimizer (default: "adam")')
parser.add_argument('--learning-rate', '--lr',
                    type=float, default=.0001, metavar='N',
                    help='learning rate of model (default: .0001)')
parser.add_argument('--sigma', '--sg',
                    type=float, default=2.0, metavar='N',
                    help='sigma for gaussian on Heatmaps (default: 2.0)')
parser.add_argument('--dataset', '--data',
                    default='coco',
                    choices=['coco'],
                    help='pick a specific dataset (default: "coco")')
parser.add_argument('--checkpoint', '--check',
                    default='none',
                    help='path to checkpoint to be restored')
parser.add_argument('--predict', '--pred',
                    action='store_true',
                    help='predict test dataset')
parser.add_argument('--plot', '--p',
                    action='store_true',
                    help='plot dataset sample')
parser.add_argument('--summary', '--sm',
                    action='store_true',
                    help='show summary of model')
args = parser.parse_args()


def batch_status(batch_idx, inputs, outputs, targets,
                 epoch, train_loader, loss, validset):
    # Global step
    global_step = batch_idx + len(train_loader) * epoch

    # update running loss statistics
    args.running_loss += loss.item()
    args.train_loss += loss.item()

    # Write tensorboard statistics
    args.writer.add_scalar('Train/loss', loss.item(), global_step)

    # print every args.log_interval of batches
    if global_step % args.log_interval == 0:
        # validate
        vloss = validate(validset, log_info=True, global_step=global_step)

        # Make targets and output slices
        trgt_htmp = heatmap(targets.sum(dim=1, keepdim=True))
        otpt_htmp = heatmap(outputs.sum(dim=1, keepdim=True))

        # Make grids
        image_grid = make_grid(inputs, nrow=4, padding=2, pad_value=1)
        trgt_htmp_grid = make_grid(trgt_htmp, nrow=4, padding=2, pad_value=1)
        otpt_htmp_grid = make_grid(otpt_htmp, nrow=4, padding=2, pad_value=1)

        # Create Heatmaps grid
        args.writer.add_image('Train/gt', trgt_htmp_grid, global_step)
        args.writer.add_image('Train/gt_image', image_grid + trgt_htmp_grid,
                              global_step)
        args.writer.add_image('Train/pred', otpt_htmp_grid, global_step)
        args.writer.add_image('Train/pred_image', image_grid + otpt_htmp_grid,
                              global_step)

        # Process current checkpoint
        process_checkpoint(loss.item(), global_step, args)

        print('Epoch : {} Batch : {} [{}/{} ({:.0f}%)]\n'
              '====> Run_Loss : {:.6f} Valid_Loss : {:.6f}'
              .format(epoch, batch_idx,
                      args.batch_size * batch_idx,
                      args.dataset_size,
                      100. * batch_idx / args.dataloader_size,
                      args.running_loss / args.log_interval,
                      vloss),
              end='\n\n')

        args.running_loss = 0.0

    # (compatibility issues) Pass all pending items to disk
    args.writer.flush()


def train(trainset, validset):
    # Create dataset loader
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               drop_last=False)
    args.dataset_size = len(train_loader.dataset)
    args.dataloader_size = len(train_loader)

    # get some random training images
    dataiter = iter(train_loader)
    batch = dataiter.next()
    images, targets = dataiter.next()

    if args.plot:
        # Create images grid
        grid = make_grid(images, nrow=4, padding=2, pad_value=1)

        # Create Heatmaps grid
        targets_slice = targets.sum(dim=1, keepdim=True)
        grid = grid + make_grid(targets_slice, nrow=4, padding=2, pad_value=1)
        imshow(grid)

        # save sample into tensorboard
        args.writer.add_image('Sample', grid)

    # Define optimizer
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam(args.net.parameters(),
                                    lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD(args.net.parameters(),
                                   lr=args.learning_rate)

    # Set loss function
    args.criterion = torch.nn.MSELoss()

    # restore checkpoint
    restore_checkpoint(args)

    # Set best for minimization
    args.best = float('inf')

    print('Started Training')
    # loop over the dataset multiple times
    for epoch in range(args.epochs):
        # reset running loss statistics
        args.train_loss = args.train_acc = args.running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader, 1):
            # Unpack batch
            inputs, targets = batch

            # Send to device
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            # Calculate gradients and update
            with autograd.detect_anomaly():
                # zero the parameter gradients
                args.optimizer.zero_grad()

                # forward
                outputs = args.net(inputs)

                # calculate loss
                loss = args.criterion(outputs, targets)

                # backward + step
                loss.backward()
                args.optimizer.step()

            # Log batch status
            batch_status(batch_idx, inputs, outputs, targets,
                         epoch, train_loader, loss, validset)

        print('Epoch: {} Average loss: {:.4f}'
              .format(epoch, args.train_loss / len(train_loader)))

    # Add trained model
    print('Finished Training')


def validate(validset, print_info=False, log_info=False, global_step=0):
    # Create dataset loader
    valid_loader = torch.utils.data.DataLoader(validset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=False)
    if print_info:
        print('Started Validation')

    run_loss = 0
    for batch_idx, batch in enumerate(valid_loader, 1):
        # Unpack batch
        inputs, targets = batch

        # Send to device
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        # Calculate gradients and update
        with autograd.detect_anomaly():
            # forward
            outputs = args.net(inputs)

            # calculate loss
            loss = args.criterion(outputs, targets)
            run_loss += loss.item()

        if batch_idx == 1:
            # Make targets and output slices
            trgt_htmp = heatmap(targets.sum(dim=1, keepdim=True))
            otpt_htmp = heatmap(outputs.sum(dim=1, keepdim=True))

            # Make grids
            image_grid = make_grid(inputs, nrow=4, padding=2, pad_value=1)
            trgt_htmp_grid = make_grid(trgt_htmp, nrow=4, padding=2, pad_value=1)
            otpt_htmp_grid = make_grid(otpt_htmp, nrow=4, padding=2, pad_value=1)

            # Create Heatmaps grid
            args.writer.add_image('Valid/gt', trgt_htmp_grid, global_step)
            args.writer.add_image('Valid/gt_image', image_grid + trgt_htmp_grid,
                                  global_step)
            args.writer.add_image('Valid/pred', otpt_htmp_grid, global_step)
            args.writer.add_image('Valid/pred_image', image_grid + otpt_htmp_grid,
                                  global_step)

    if log_info:
        args.writer.add_scalar('Valid/loss',
                               run_loss / len(valid_loader),
                               global_step)

    return run_loss / len(valid_loader)


def predict_test(testset):
    # Create dataset loader
    # Create dataset loader
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              drop_last=False)

    # restore checkpoint
    restore_checkpoint(args)

    # Set loss function
    args.criterion = loss_yolo(args).measure

    # Predict all test elements and measure
    run_loss = 0
    for batch_idx, batch in enumerate(test_loader, 1):
        # Unpack batch
        inputs, targets = batch

        # Send to device
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        # Calculate gradients and update
        with autograd.detect_anomaly():
            # forward
            outputs = args.net(inputs)

            # calculate loss
            loss, t_outputs = args.criterion(outputs, targets)
            run_loss += loss.item()

        if batch_idx < 10:
            # Plot predictions
            img = imshow_bboxes(inputs, targets, args, t_outputs)
            args.writer.add_image('Test/predicted', img, batch_idx)
        else:
            break


def main():
    # Printing parameters
    torch.set_printoptions(precision=2)
    torch.set_printoptions(edgeitems=5)

    # Set up GPU
    if args.device != 'cpu':
        args.device = torch.device('cuda:0'
                                   if torch.cuda.is_available()
                                   else 'cpu')

    # Selected device for trainning or inference
    print('device : {}'.format(args.device))

    # Read parameters from checkpoint
    if args.checkpoint:
        read_checkpoint(args)

    # Save parameters in string to name the execution
    args.run = create_run_name(args)

    # print run name
    print('execution name : {}'.format(args.run))

    # if not args.predict:
    # Tensorboard summary writer
    writer = SummaryWriter('runs/' + args.run)

    # Save as parameter
    args.writer = writer

    # Read dataset
    if args.dataset == 'coco':
        trn, vld = load_dataset(args)

    # Get hparams from args
    args.hparams = get_hparams(args.__dict__)
    print('\nParameters :')
    pprint(args.hparams)
    print()

    # Create network
    if args.network == 'posture':
        net = Resnet_Posture(args)

        # Load pretrained resnet weights. Drop gradients.
        for param in net.resnet.parameters():
            param.requires_grad = False

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

    # (compatibility issues) Add hparams with metrics to tensorboard
    # args.writer.add_hparams(args.hparams, {'metrics': 0})

    # Delete model + Free memory
    del args.net
    torch.cuda.empty_cache()

    # if not args.predict:
    # Close tensorboard writer
    args.writer.close()


if __name__ == "__main__":
    main()
