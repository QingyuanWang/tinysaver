# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from dyce import OutputRecorder
import math
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from datasets import build_dataset, build_loader
from models.dmc import DMC_train
import os
import glob


def compute_loss(output, ee_out, targets, normlized_macs_ratio):
    if targets.shape[-1] == 1:
        ee_loss_raw_ce = F.cross_entropy(
            ee_out[:, 1:, :].permute(0, 2, 1), targets.expand(-1, ee_out.shape[1] - 1), reduction='none'
        )
    else:
        ee_loss_raw_ce = F.cross_entropy(
            ee_out[:, 1:, :].permute(0, 2, 1),
            targets.unsqueeze(-1).expand(-1, -1, ee_out.shape[1] - 1),
            reduction='none'
        )
    conf = ee_out[:, 0, :].softmax(dim=-1).amax(dim=-1)
    ee_loss_raw = (((1 - conf) * 2).unsqueeze(-1) * ee_loss_raw_ce).mean(dim=0)
    ee_loss = ee_loss_raw.sum()  #/ ee_loss_raw[1:].shape[0]
    ori_loss = F.cross_entropy(output, targets.squeeze(-1))
    ori_loss = ori_loss.mean()
    return ee_loss, ori_loss, ee_loss_raw


def compute_ild_loss(feat_seq):
    loss_raw = F.mse_loss(
        feat_seq[:, :-1, :],
        feat_seq[:, -1, :].detach().unsqueeze(1).expand(-1, feat_seq.shape[1] - 1, -1),
        reduction='none'
    ).mean(dim=(0, 2))
    return loss_raw.sum(), loss_raw


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    accelerator,
    normlized_macs_ratio: torch.Tensor,
    model_ema: Optional[ModelEma] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    mixup_fn: Optional[Mixup] = None,
    transform_gpu: Optional[callable] = None,
    log_writer=None,
    args=None
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('ee_acc', utils.SmoothedVector())
    metric_logger.add_meter('ee_loss_raw', utils.SmoothedVector(monitor=False))
    # metric_logger.add_meter('ild_loss_raw', utils.SmoothedVector(monitor=False))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    update_freq = args.update_freq
    use_amp = args.use_amp
    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if lr_scheduler is not None:
            lr_scheduler.step(data_iter_step / len(data_loader) + epoch)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(-1)
        if transform_gpu is not None:
            samples = transform_gpu(samples)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=use_amp):
            output, ee_out = model(samples)
            ee_loss, ori_loss, ee_loss_raw = compute_loss(output, ee_out, targets, normlized_macs_ratio)
            # ild_loss, ild_loss_raw = compute_ild_loss(feat_seq)

        loss = ee_loss  #+ ild_loss
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        loss /= update_freq
        # loss.backward()
        accelerator.backward(loss)
        # if accelerator.sync_gradients:
        #     accelerator.clip_grad_norm_(model.parameters(), 0.1)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.step()
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

        # torch.cuda.synchronize()

        if mixup_fn is not None:
            targets = targets.argmax(-1).unsqueeze(-1)
        ee_acc = (ee_out.argmax(-1) == targets).float().mean(dim=0)
        metric_logger.update(ee_loss=ee_loss.item())
        metric_logger.update(ee_loss_raw=ee_loss_raw)
        # metric_logger.update(ild_loss=ild_loss.item())
        # metric_logger.update(ild_loss_raw=ild_loss_raw)
        ori_acc = (output.argmax(-1) == targets.squeeze(-1)).float().mean()
        ee_acc = torch.cat([ee_acc, ori_acc.unsqueeze(-1)], dim=0)
        metric_logger.update(ee_acc=ee_acc)
        metric_logger.update(ori_loss=ori_loss.item())
        metric_logger.update(ori_acc=ori_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        if log_writer is not None and data_iter_step % 1000 == 0:
            log_writer.update(ee_loss=ee_loss.item(), head="loss")
            # log_writer.update(ild_loss=ild_loss.item(), head="loss")
            log_writer.update_hist(ee_acc=ee_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            if epoch == 0:
                log_writer.update(ori_loss=ori_loss.item(), head="loss")
                log_writer.update(ori_acc=ori_acc, head="loss")
            log_writer.set_step()
        # torch.distributed.barrier()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, accelerator, normlized_macs_ratio, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('ee_acc', utils.SmoothedVector())
    metric_logger.add_meter('ee_loss_raw', utils.SmoothedVector())
    # metric_logger.add_meter('ild_loss_raw', utils.SmoothedVector())
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        targets = batch[-1]

        images = images.to(accelerator.device, non_blocking=True)
        targets = targets.to(accelerator.device, non_blocking=True).unsqueeze(-1)
        # compute output
        output, ee_out = model(images)
        ee_loss, ori_loss, ee_loss_raw = compute_loss(output, ee_out, targets, normlized_macs_ratio)
        # ild_loss, ild_loss_raw = compute_ild_loss(feat_seq)

        output, targets, ee_out = accelerator.gather_for_metrics((output, targets, ee_out))

        # torch.cuda.synchronize()
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        # ori_acc = (output.max(-1)[-1] == targets).float().mean()
        ee_acc = (ee_out.argmax(-1) == targets).float().mean(dim=0)

        batch_size = output.shape[0]
        metric_logger.meters['ori_acc'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['ori_loss'].update(ori_loss.item(), n=batch_size)
        metric_logger.meters['ee_acc'].update(ee_acc, n=batch_size)
        metric_logger.meters['ee_loss'].update(ee_loss.item(), n=batch_size)
        metric_logger.meters['ee_loss_raw'].update(ee_loss_raw, n=batch_size)
    print(
        '* OriAcc {ori_acc.global_avg:.3f} Loss {ori_loss.global_avg:.3f} EEloss{ee_loss.global_avg:.3f} '.format(
            ori_acc=metric_logger.ori_acc, ori_loss=metric_logger.ori_loss, ee_loss=metric_logger.ee_loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def record_prediction(
    data_loader,
    model,
    device,
    save_path,
    topk=1,
    ckpt_type_config=None,
    transform_gpu: Optional[callable] = None,
):
    # switch to evaluation mode
    model.eval()
    recorder = OutputRecorder(topk)
    logger = utils.MetricLogger(delimiter="  ")
    for batch in logger.log_every(data_loader, 200, header='record_prediction'):
        images = batch[0]
        targets = batch[-1]

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(-1)
        if transform_gpu is not None:
            images = transform_gpu(images)
        # compute output
        model_out = model(images)
        if isinstance(model_out, tuple):
            recorder.add_pred(model_out[0], model_out[1])
        else:
            recorder.add_pred(model_out)
        recorder.add_label(targets)

    return recorder.save(save_path, ckpt_type_config)
