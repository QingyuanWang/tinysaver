# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import numpy as np
import time
import json
import os
from pathlib import Path
from torch import optim as optim
import torch
import torch.backends.cudnn as cudnn

from timm.data.mixup import Mixup
from timm.utils import ModelEmaV2

from datasets import build_dataset, build_loader, MultiEpochsDataLoaderWrapper
from engine import train_one_epoch, evaluate, record_prediction

import utils
from utils import get_args_parser, build_model, setup_for_distributed, load_model
from utils import CosineAnnealingWarmUp
import models.dmc as dmc
from accelerate import Accelerator
import subprocess
from macs_analysis import get_complexity
import sys
import signal


def main(args, accelerator):

    device = torch.device(accelerator.device)
    torch.set_float32_matmul_precision('high')
    args.distributed = True

    # utils.init_distributed_mode(args)
    setup_for_distributed(accelerator.local_process_index == 0)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    dataset_train, args.nb_classes, transform_gpu = build_dataset(is_train=True, args=args)

    dataset_val, _, _ = build_dataset(is_train=False, args=args)

    if args.debug:
        print(f'Debug mode is activated, dataset size is reduced to 1/1000')
        dataset_val = torch.utils.data.Subset(dataset_val, range(len(dataset_val) // 1000))
        dataset_train = torch.utils.data.Subset(dataset_train, range(len(dataset_train) // 1000))
    print(f'Dataset size: {len(dataset_train)}, {len(dataset_val)}')

    data_loader_train, data_loader_val = build_loader(dataset_train, dataset_val, args)

    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        if args.tensorboard_dir == 'default':
            args.tensorboard_dir = f'{args.output_dir}/tensorboard'
        Path(args.record_dir).mkdir(parents=True, exist_ok=True)
        Path(args.tensorboard_dir).mkdir(parents=True, exist_ok=True)
        log_writer = None  #utils.TensorboardLogger(log_dir=args.tensorboard_dir)
        try:
            print(subprocess.check_output(['nvidia-smi']).decode('utf-8'))

        except:
            print('Fail to execute nvidia-smi')
        args.time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
            if not commit.startswith('fatal') and not 'not found' in commit:
                args.commit = commit
        except:
            args.commit = 'not found'
        with open(os.path.join(args.output_dir, "training.log"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(vars(args), indent=4) + "\n")
    else:
        log_writer = None
    print(args)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes
        )

    model_base = build_model(args.model, accelerator.device, interm_feat=True)
    model_saver = build_model(args.saver_model, accelerator.device, interm_feat=True)
    model_base = torch.compile(model_base, dynamic=False, fullgraph=True)
    model_saver = torch.compile(model_saver, dynamic=False, fullgraph=True)
    model = dmc.DMC_train(
        model_base,
        model_saver,
        num_classes=args.nb_classes,
        input_size=(3, args.input_size, args.input_size),
        dim=args.ee_dim,
        num_attn_layer=args.num_attn_layer,
        allowed_exits=eval(args.allowed_exits),
    )
    print('DMC init finish')
    model = torch.compile(model, dynamic=False)
    model.to(device)

    if args.model_ema:
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay)
    else:
        model_ema = None
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable_params = list(filter(lambda x: x.requires_grad, [p for p in model.parameters()]))
    optimizer = optim.__dict__[args.opt](trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    # loss_scaler = NativeScaler()
    utils.auto_load_model(args=args, model=model, optimizer=optimizer, model_ema=model_ema)
    scheduler = CosineAnnealingWarmUp(optimizer, args.warmup_epochs, args.epochs, args.min_lr)
    model, optimizer, data_loader_train, scheduler = accelerator.prepare(model, optimizer, data_loader_train, scheduler)
    data_loader_val = accelerator.prepare(data_loader_val)
    if args.dataloader == 'multiepoch':
        data_loader_train = MultiEpochsDataLoaderWrapper(data_loader_train)
        data_loader_val = MultiEpochsDataLoaderWrapper(data_loader_val)

    # macs_dict = get_complexity(args.model, args.saver_model, dim=args.ee_dim)
    # normlized_macs_ratio = torch.tensor(
    #     macs_dict['backbone']['macs_ratio'][1:-1], device=device, requires_grad=False
    # ) / 100
    normlized_macs_ratio = None

    print("Model = %s" % str(model))
    eff_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // eff_batch_size
    print("lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.update_freq)
    print("effective batch size: %d" % eff_batch_size)

    trainable_params_str = 'trainable_params:\n'
    trainable_params_str += '=' * 100
    trainable_params_str += f'\n{"Name":<50}{"Number":>20}{"Shape":>20}\n'
    for name, params in model.named_parameters():
        if params.requires_grad:
            trainable_params_str += f'{name:<50}{params.numel():>20,d}{str(params.shape).removeprefix("torch.Size(").removesuffix(")"):>20}\n'
    trainable_params_str += '=' * 100
    trainable_params_str += f'\nTrainable params:{n_parameters:,d}\n'
    print(trainable_params_str)
    # =========================================
    if args.eval:
        print(f"Eval only mode")

    else:
        max_accuracy = 0.0

        print("Start training for %d epochs" % args.epochs)
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            # if args.distributed and not args.webds:
            #     data_loader_train.sampler.set_epoch(epoch)
            if log_writer is not None:
                log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
            train_stats = train_one_epoch(
                model,
                data_loader_train,
                optimizer,
                device,
                epoch,
                accelerator,
                normlized_macs_ratio,
                lr_scheduler=scheduler,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
                transform_gpu=transform_gpu,
                log_writer=log_writer,
                args=args
            )
            if accelerator.is_main_process and args.output_dir and args.save_ckpt:
                if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                    utils.save_model(
                        args=args,
                        model=model,
                        optimizer=optimizer,
                        #  loss_scaler=loss_scaler,
                        epoch=epoch,
                        model_ema=model_ema
                    )
            torch.distributed.barrier()
            if args.eval_interval != 0 and (epoch + 1) % args.eval_interval == 0 and data_loader_val is not None:
                test_stats = evaluate(data_loader_val, model, accelerator, normlized_macs_ratio, use_amp=args.use_amp)
                print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['ori_acc']:.1f}%")
                print(f'Max accuracy: {max_accuracy:.2f}%')

                if log_writer is not None:
                    log_writer.update(test_ori_acc=test_stats['ori_acc'], head="perf", step=epoch)
                    # log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                    log_writer.update(test_loss=test_stats['ori_loss'], head="perf", step=epoch)
                    log_writer.update(test_ee_loss=test_stats['ee_loss'], head="perf", step=epoch)

                    log_writer.update_hist(ee_acc=test_stats['ee_acc'], head="perf", step=epoch)
                test_stats['ee_acc'] = np.array_str(test_stats['ee_acc'], max_line_width=200, precision=4)
                train_stats['ee_acc'] = np.array_str(train_stats['ee_acc'], max_line_width=200, precision=4)
                test_stats['ee_loss_raw'] = np.array_str(test_stats['ee_loss_raw'], max_line_width=200, precision=4)
                train_stats['ee_loss_raw'] = np.array_str(train_stats['ee_loss_raw'], max_line_width=200, precision=4)
                # test_stats['ild_loss_raw'] = np.array_str(test_stats['ild_loss_raw'],max_line_width=200,precision=4)
                # train_stats['ild_loss_raw'] = np.array_str(train_stats['ild_loss_raw'],max_line_width=200,precision=4)
                log_stats = {
                    **{
                        f'train_{k}': v for k, v in train_stats.items()
                    },
                    **{
                        f'test_{k}': v for k, v in test_stats.items()
                    }, 'epoch': epoch,
                    'n_parameters': n_parameters
                }

            else:
                train_stats['ee_acc'] = np.array_str(train_stats['ee_acc'], max_line_width=200, precision=4)
                train_stats['ee_loss_raw'] = np.array_str(train_stats['ee_loss_raw'], max_line_width=200, precision=4)
                log_stats = {
                    **{
                        f'train_{k}': v for k, v in train_stats.items()
                    }, 'epoch': epoch,
                    'n_parameters': n_parameters
                }
                # train_stats['ild_loss_raw'] = str(train_stats['ild_loss_raw'],max_line_width=200,precision=4)

            if args.output_dir and accelerator.is_main_process:
                if log_writer is not None:
                    log_writer.flush()

                log_stats['time'] = args.time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(os.path.join(args.output_dir, "training.log"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats, indent=4) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
    # =========================================
    # Save inference records on training/val set
    if args.record_pred and accelerator.is_main_process:
        data_loader_train.close()
        data_loader_val.close()
        del optimizer
        del model
        del model_ema
        del data_loader_train
        del data_loader_val
        del scheduler
        print('Finish resource clean up')
        produce_record(args, accelerator)


def produce_record(args, accelerator):
    print('Start producing records')
    device = torch.device(accelerator.device)
    args.shuffle = False
    args.lmdb = False
    args.webds = False
    model_base = build_model(args.model, accelerator.device, interm_feat=True)
    model_saver = build_model(args.saver_model, accelerator.device, interm_feat=True)
    # model_base = torch.compile(model_base, dynamic=False, fullgraph=True)
    # model_saver = torch.compile(model_saver, dynamic=False, fullgraph=True)
    model = dmc.DMC_train(
        model_base,
        model_saver,
        num_classes=args.nb_classes,
        input_size=(3, args.input_size, args.input_size),
        dim=args.ee_dim,
        num_attn_layer=args.num_attn_layer,
        allowed_exits=eval(args.allowed_exits),
    )
    print('DMC init finish')
    model.to(device)
    dataset_train, args.nb_classes, transform_gpu = build_dataset(eval_mode=False, is_train=True, args=args)
    dataset_val, args.nb_classes, _ = build_dataset(eval_mode=False, is_train=False, args=args)
    if args.debug:
        print(f'Debug mode is activated, dataset size is reduced to 1/1000')
        dataset_train = torch.utils.data.Subset(dataset_train, range(len(dataset_train) // 1000))
        dataset_val = torch.utils.data.Subset(dataset_val, range(len(dataset_val) // 1000))
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_train = torch.utils.data.SequentialSampler(dataset_train)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )
    # model = model._orig_mod
    # load_model(args=args, model=model, ema=True)
    # model = torch.compile(model, dynamic=False)

    record_prediction(data_loader_val, model, device, f'{args.record_dir}/{args.sim_index}_test')
    # load_model(args=args, model=model,ema=False)

    record_prediction(
        data_loader_train, model, device, f'{args.record_dir}/{args.sim_index}_train', transform_gpu=transform_gpu
    )
    torch.distributed.barrier()
    print('Job finished, kill itself')
    os.kill(os.getpid(), signal.SIGKILL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TinySaver Training', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.sim_index)
    args.record_dir = os.path.join(args.record_dir, args.sim_index)
    if os.path.exists(f'{args.record_dir}/{args.sim_index}_train.npz'
                     ) and os.path.exists(f'{args.record_dir}/{args.sim_index}_test.npz') and not args.force_refresh:
        print(f'./records/{args.sim_index} exists, abort.')
        sys.exit()
    accelerator = Accelerator()
    if args.skip_training or os.path.exists(f'{args.output_dir}/checkpoint-{args.epochs-1}.pth'):
        print('Skip training')
        produce_record(args, accelerator)
    else:
        main(args, accelerator)
