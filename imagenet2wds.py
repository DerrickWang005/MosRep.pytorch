import os
import os.path as osp
import torch
import random
import argparse
import webdataset as wds
import torchvision.datasets as datasets


parser = argparse.ArgumentParser(description='Webdataset convertion')
parser.add_argument('-i', '--data-dir', metavar='DIR', help='path to imagenet')
parser.add_argument('-o', '--output-dir', metavar='DIR', help='path to output folder')


def readfile(fname):
    with open(fname, "rb") as stream:
        return stream.read()


def wds_shards_create(args):
    trainset = datasets.ImageNet(args.data_dir, split='train')
    valset = datasets.ImageNet(args.data_dir, split='val')
    print(len(trainset), len(valset))
    os.makedirs(osp.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(osp.join(args.output_dir, 'val'), exist_ok=True)
    print('init dataset')

    train_keys = set()
    val_keys = set()

    indexes = list(range(len(trainset.imgs)))
    random.shuffle(indexes)
    with wds.ShardWriter(osp.join(args.output_dir, 'train', 'shards-%05d.tar'), maxcount=1000) as sink:
        for i in indexes:
            fname, cls = trainset.imgs[i]
            assert cls == trainset.targets[i]
            image = readfile(fname)

            key = os.path.splitext(os.path.basename(fname))[0]
            assert key not in train_keys
            train_keys.add(key)

            sample = {"__key__": str(i), "jpg": image, "cls": cls}
            sink.write(sample)
    print('finish trainset')

    indexes = list(range(len(valset.imgs)))
    random.shuffle(indexes)
    with wds.ShardWriter(osp.join(args.output_dir, 'val', 'shards-%05d.tar'), maxcount=1000) as sink:
        for i in indexes:
            fname, cls = valset.imgs[i]
            assert cls == valset.targets[i]
            image = readfile(fname)

            key = os.path.splitext(os.path.basename(fname))[0]
            assert key not in val_keys
            val_keys.add(key)

            sample = {"__key__": str(i), "jpg": image, "cls": cls}
            sink.write(sample)
    print('finish valset')


if __name__ == '__main__':
    args = parser.parse_args()
    wds_shards_create(args)
