# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision import datasets
import torchvision.transforms.v2 as transforms
import torch
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


class MultiEpochsDataLoaderWrapper:
    '''
        A moddified wrapper of timm.data.loader.MultiEpochsDataLoader to support accelerator's DataLoaderShard.
        This wrapper can reduce the data loading time on first few batchs of every epoch.
    '''

    def __init__(self, dataloader):
        self.dl = dataloader
        self.length = len(self.dl)
        self.iterator = self.dl.__iter__()

    def __len__(self):
        return self.length

    def __iter__(self):
        self.dl.begin()
        for i in range(len(self)):
            yield next(self.iterator)
        self.dl.iteration += 1
        self.dl.end()
    
    def close(self):
        del self.dl

class RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    @property
    def batch_size(self):
        return self.sampler.batch_size
    @property
    def drop_last(self):
        return self.sampler.drop_last

    def __len__(self):
        return len(self.sampler)


    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def build_dataset(is_train, args, eval_mode=False):
    transform, transform_batch = build_transform(False if eval_mode else is_train, args)
    # transform = torch.compile(transform)
    print(f"Transform (is_train={is_train}) = ")
    print(transform)
    if transform_batch is not None:
        print("Transform batch = ")
        print(transform_batch)
    print("---------------------------")
    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        dataset = datasets.ImageNet(args.data_path, transform=transform,
                                    split='train' if is_train else 'val')
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        print("Number of the class = %d" % nb_classes)
        print(len(dataset.class_to_idx))
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)
    if isinstance(dataset, torch.utils.data.Dataset):
        print("Shape of each image", dataset[0][0].shape)
    print("Number of images", len(dataset))
    return dataset, nb_classes, transform_batch


def build_loader(dataset_train, dataset_val, args):


    sampler_train = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(dataset_train),
                                                  args.batch_size,
                                                  drop_last=True)
    sampler_val = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(dataset_val),
                                                args.batch_size,
                                                drop_last=False)
    if args.dataloader=='multiepoch':
        sampler_train = RepeatSampler(sampler_train)
        sampler_val = RepeatSampler(sampler_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        # sampler=sampler_train,
        # batch_size=args.batch_size,
        batch_sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        # shuffle=True,
        # drop_last=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor
    )
    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            # batch_size=args.batch_size,
            num_workers=args.num_workers,
            # shuffle=False,
            batch_sampler=sampler_val,
            pin_memory=args.pin_mem,
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor
        )
    else:
        data_loader_val = None
    return data_loader_train, data_loader_val


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    print('Resize image: ', resize_im)
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
    if args.data_set == 'CIFAR10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        if args.gpuaug:
            transform = torch.nn.Sequential(
                transforms.ToImage(),
                transforms.ToDtype(torch.uint8, scale=True),
                transforms.RandomResizedCrop(args.input_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            )
            transform_gpu = torch.nn.Sequential(
                transforms.RandomHorizontalFlip(), transforms.AutoAugment(), transforms.ToDtype(torch.float32, scale=True), transforms.Normalize(mean, std)
            )
            return transform, transform_gpu
        else:
            transform = torch.nn.Sequential(
                transforms.ToImage(), transforms.ToDtype(torch.uint8, scale=True),
                transforms.RandomResizedCrop(args.input_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.RandomHorizontalFlip(), transforms.AutoAugment(), transforms.ToDtype(torch.float32, scale=True), transforms.Normalize(mean, std)
            )
            return transform, None
    else:
        t = [transforms.ToImage(), transforms.ToDtype(torch.uint8, scale=True)]
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:
            t.append(transforms.Resize((args.input_size, args.input_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True), )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToDtype(torch.float32, scale=True))
    t.append(transforms.Normalize(mean, std))
    return torch.nn.Sequential(*t), None
