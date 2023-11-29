from __future__ import print_function

import argparse
import os
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from dataset.cifar10 import get_cifar10_dataloaders
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.imagenet16 import get_imagenet16_dataloaders
from helper.loops import train_vanilla as train
from helper.loops import validate
from helper.util import AverageMeter, accuracy, adjust_learning_rate
from models import model_dict


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq',
                        type=int,
                        default=100,
                        help='print frequency')
    parser.add_argument('--tb_freq',
                        type=int,
                        default=500,
                        help='tb frequency')
    parser.add_argument('--save_freq',
                        type=int,
                        default=40,
                        help='save frequency')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='batch_size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs',
                        type=int,
                        default=240,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs',
                        type=str,
                        default='150,180,210',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate',
                        type=float,
                        default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet110')
    parser.add_argument('--dataset',
                        type=str,
                        default='cifar100',
                        choices=['cifar100', 'cifar10', 'imagenet16'],
                        help='dataset')

    parser.add_argument('-t',
                        '--trial',
                        type=int,
                        default=0,
                        help='the experiment id')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = f'./save/models/{opt.dataset}'
        opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(
        opt.model, opt.dataset, opt.learning_rate, opt.weight_decay, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():
    best_acc = 0

    opt = parse_option()

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(
            batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    elif opt.dataset == 'cifar10':
        train_loader, val_loader = get_cifar10_dataloaders(
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            is_instance=False)
        n_cls = 10
    elif opt.dataset == 'imagenet16':
        train_loader, val_loader = get_imagenet16_dataloaders(
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
        )
        n_cls = 120
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = model_dict[opt.model](num_classes=n_cls)

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print('==> training...')

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion,
                                      optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model,
                                                      criterion, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder,
                                     '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
