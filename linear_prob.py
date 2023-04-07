#!/usr/bin/env python
import argparse
import math
import os
import random
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from loguru import logger

import moco.resnet as models
import utils.misc as misc
from utils.dataset import (get_linear_transform, get_test_transform,
                           get_wds_dataset, LinearTransform)
from utils.lars import LARC

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--train-data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--val-data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--train-num-samples', default=1281167, type=int,
                    help='number of training samples (default: 1281167)')
parser.add_argument('--val-num-samples', default=50000, type=int,
                    help='number of validation samples (default: 50000)')
parser.add_argument("--train-data-upsampling-factors",
                    type=str,
                    default=None,
                    help=(
                        "When using multiple data sources with webdataset and sampling with replacement,"
                        "this can be used to upsample specific data sources. "
                        "Similar to --train-data, this should be a string with as many numbers as there are data sources,"
                        "separated by `::` (e.g. 1::2::0.5) "
                        "By default, datapoints are sampled uniformly regardless of the dataset sizes."
                    ))
parser.add_argument("--dataset-resampled",
                    default=False,
                    action="store_true",
                    help="Whether to use sampling with replacement for webdataset shard selection."
)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--num-classes', default=1000, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--dist-url', default='tcp://localhost:1681', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--exp-folder', default='exp', type=str,
                    help='experiment Folder.')
parser.add_argument('--exp-name', default='exp_name', type=str,
                    help='experiment Name.')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')

best_acc1 = 0


@logger.catch
def main():
    global best_acc1

    # configuration
    args = parser.parse_args()

    # cuda & cudnn backends
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # intialize distributed device environment
    device = misc.init_distributed_device(args)

    # random seed
    if args.seed is not None:
        np.random.seed(args.seed + args.rank)
        random.seed(args.seed + args.rank)
        torch.manual_seed(args.seed + args.rank)

    # output folder
    date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    if args.distributed:
        # sync date_str from master to all ranks
        date_str = misc.broadcast_object(args, date_str)
    args.exp_name = '{}-{}-{}'.format(date_str, args.exp_name, 'linear_prob')
    args.output_dir = os.path.join(args.exp_folder, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # setup logger
    misc.setup_logger(
        args.output_dir,
        distributed_rank=args.rank,
        filename="linear.log",
        mode="a")

    # create model
    logger.info("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=args.num_classes)
    logger.info(model)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # load from pre-trained model
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            logger.info("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q.') and not k.startswith('module.encoder_q.fc.'):
                    new_state_dict[k.replace('module.encoder_q.', '')] = state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(new_state_dict, strict=False)
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            logger.info(msg)
            logger.info("=> loaded pre-trained model '{}'".format(args.pretrained))
            del state_dict, new_state_dict
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.distributed:
        total_batch_size = args.world_size * args.batch_size
        args.lr = args.lr * (total_batch_size / 256.)
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.local_rank)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(
        parameters, args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.distributed:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.local_rank)
                checkpoint = torch.load(args.resume, map_location=loc)
            else:
                checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.distributed:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.local_rank)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    preprocess_train = LinearTransform(get_linear_transform(args))
    preprocess_val = LinearTransform(get_test_transform(args))
    train_data = get_wds_dataset(args, preprocess_train, 'train', args.start_epoch)
    val_data = get_wds_dataset(args, preprocess_val, 'val', args.start_epoch)
    logger.info(train_data.dataloader.num_samples)
    logger.info(val_data.dataloader.num_samples)

    # only evaluation
    if args.evaluate:
        validate(val_data.dataloader, model, criterion, args)
        return

    # start training
    dist.barrier()
    start = datetime.now()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_data.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_data.dataloader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_data.dataloader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if misc.is_master(args):
            misc.save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, output_dir=args.output_dir, filename='latest.pth.tar')
            # if epoch == args.start_epoch:
            #     try:
            #         misc.sanity_check(model.state_dict(), args.pretrained)
            #     except:
            #         pass
    logger.info(f"Max Accuracy={best_acc1}")
    total_seconds = (datetime.now() - start).total_seconds()
    logger.info('Total seconds: {}'.format(total_seconds))
    dist.destroy_process_group()


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = misc.AverageMeter('Time', ':3.3f')
    data_time = misc.AverageMeter('Data', ':3.3f')
    losses = misc.AverageMeter('Loss', ':.4f')
    top1 = misc.AverageMeter('Acc@1', ':3.2f')
    top5 = misc.AverageMeter('Acc@5', ':3.2f')
    progress = misc.ProgressMeter(
        train_loader.num_batches,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(args.local_rank, non_blocking=True)
        target = target.cuda(args.local_rank, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = misc.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = misc.AverageMeter('Time', ':3.3f')
    losses = misc.AverageMeter('Loss', ':.4f')
    top1 = misc.AverageMeter('Acc@1', ':3.2f')
    top5 = misc.AverageMeter('Acc@5', ':3.2f')
    progress = misc.ProgressMeter(
        val_loader.num_batches,
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.local_rank, non_blocking=True)
            target = target.cuda(args.local_rank, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = misc.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return top1.avg


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group["lr"] = cur_lr


if __name__ == '__main__':
    main()
