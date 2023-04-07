#!/usr/bin/env python
import argparse
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
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from loguru import logger
from torch.cuda.amp import GradScaler, autocast

import moco.resnet as models
import utils.misc as misc
from moco.builder import MoCo, MosRep
from utils.dataset import (get_local_transform, get_standard_transform,
                           get_wds_dataset, CropsTransform)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--train-data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--train-num-samples', default=1281167, type=int,
                    help='number of training samples (default: 1281167)')
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
parser.add_argument(
    "--dataset-resampled",
    default=False,
    action="store_true",
    help="Whether to use sampling with replacement for webdataset shard selection."
)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-m', '--method', default="moco", type=str,
                    help='type of methods (moco, mosaic)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('-s', '--save-freq', default=50, type=int,
                    metavar='N', help='save frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:4681', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--exp-folder', default='FOLDER_DIR', type=str,
                    help='experiment folder.')
parser.add_argument('--exp-name', default='EXP_NAME', type=str,
                    help='experiment name')

# moco specific configs:
parser.add_argument('--cos', action='store_true',
                    help='cosine annealing lr decay')
parser.set_defaults(cos=True)
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=16384, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--multi-crop', default=False, action="store_true",
                    help="Use multi-crop augmentation (default: False)")
parser.add_argument('--global-size', type=int, default=224,
                    help="""Input size of global crop""")
parser.add_argument('--local-size', type=int, default=112,
                    help="""Input size of local crop""")
parser.add_argument('--global-scale', type=float, nargs='+', default=(0.2, 1.),
                    help="""Scale range of the cropped image before resizing,
                    relatively to the origin image. Used for large global view cropping.
                    We recommand using a wider range of scale (0.14~1.)""")
parser.add_argument('--local-scale', type=float, nargs='+', default=(0.05, 0.14),
                    help="""Scale range of the cropped image before resizing,
                    relatively to the origin image. Used for small local view cropping of multi-crop.""")
parser.add_argument('--shift-enable', default=1.0, type=float,
                    help='the percentage of shifting')
parser.add_argument('--shift-pix', default=48, type=int,
                    help='the range of shfiting pixels')
parser.add_argument('--shift-beta', default=0.5, type=float,
                    help='the shifting beta distribution')


@logger.catch
def main():
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
    args.exp_name = '{}-{}'.format(date_str, args.exp_name)
    args.output_dir = os.path.join(args.exp_folder, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # setup logger
    misc.setup_logger(
        args.output_dir,
        distributed_rank=args.rank,
        filename="pretrain.log",
        mode="a")

    # create model
    logger.info("=> {}: creating model '{}'".format(args.method, args.arch))
    if args.method == 'mocov2':
        model = MoCo(
            base_encoder=models.__dict__[args.arch],
            dim=args.moco_dim,
            K=args.moco_k,
            m=args.moco_m,
            T=args.moco_t,
        )
    elif args.method == 'mosrep':
        model = MosRep(
            base_encoder=models.__dict__[args.arch],
            dim=args.moco_dim,
            K=args.moco_k,
            m=args.moco_m,
            T=args.moco_t,
            shift_enable=args.shift_enable,
            shift_pix=args.shift_pix,
            shift_beta=args.shift_beta)
    model = model.cuda(args.local_rank)
    logger.info(model)

    if args.distributed:
        total_batch_size = args.world_size * args.batch_size
        args.lr = args.lr * (total_batch_size / 256.)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # optimizer & amp scaler
    parameters = model.parameters()
    optimizer = torch.optim.SGD(
        parameters, args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    scaler = GradScaler()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")

    # Data loading
    preprocess = CropsTransform(
        transform_standard=get_standard_transform(args),
        transform_local=get_local_transform(args),
        multi_crop=args.multi_crop
    )
    train_data = get_wds_dataset(args, preprocess, 'pretrain', args.start_epoch)
    logger.info(train_data.dataloader.num_samples)

    # start training
    dist.barrier()
    start = datetime.now()
    for epoch in range(args.start_epoch, args.epochs):
        # set epoch in process safe manner via shared_epoch
        if args.distributed:
            train_data.set_epoch(epoch)

        # adjust lr by epoch
        misc.adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_data.dataloader, model, optimizer, scaler, epoch, args)

        # save checkpoint
        if misc.is_master(args) and (epoch + 1) % args.save_freq == 0:
            logger.info(f"* {epoch}ep save at {args.output_dir} *")
            misc.save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, output_dir=args.output_dir, filename='pretrain_{}ep.pth.tar'.format(epoch+1))

    total_seconds = (datetime.now() - start).total_seconds()
    logger.info('Total seconds: {}'.format(total_seconds))
    dist.destroy_process_group()


def train(train_loader, model, optimizer, scaler, epoch, args):
    batch_time = misc.AverageMeter('Time', ':3.3f')
    data_time = misc.AverageMeter('Data', ':3.3f')
    lr = misc.AverageMeter('Lr', ':1.5f')
    losses_s = misc.AverageMeter('Loss@S', ':.4f')
    losses_m = misc.AverageMeter('Loss@M', ':.4f')
    losses = misc.AverageMeter('Loss@Total', ':.4f')
    top_s = misc.AverageMeter('Acc@S', ':3.2f')
    top_m = misc.AverageMeter('Acc@M', ':3.2f')
    progress = misc.ProgressMeter(
        train_loader.num_batches,
        [batch_time, data_time, lr, losses_s, losses_m, losses, top_s, top_m],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    # switch to train mode
    model.train()
    time.sleep(1)
    end = time.time()
    for i, batch in enumerate(train_loader):
        # two views:
        #   query: B, 3, H, W
        #   key: B, 3, H, W
        # two views & multi crops
        #   query: B, 3, H, W
        #   query_mini: B, 4, 3, H, W
        #   key: B, 3, H, W
        batch = [item.cuda(args.local_rank, non_blocking=True) for item in batch]
        B = batch[0].size(0)
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        with autocast():  # automatic mixed precision
            if args.multi_crop:
                loss_single, loss_mosaic, acc_single, acc_mosaic = model(batch, args.multi_crop)
                loss = loss_single + loss_mosaic
            else:
                loss_single, acc_single = model(batch)
                loss = loss_single

        # measure accuracy and record loss
        losses_s.update(loss_single.item(), B)
        losses.update(loss.item(), B)
        top_s.update(acc_single, B)
        if args.multi_crop:
            losses_m.update(loss_mosaic.item(), B)
            top_m.update(acc_mosaic, B)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()

        # measure elapsed time
        lr.update(optimizer.param_groups[0]["lr"])
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    main()
