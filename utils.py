# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import time
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict

from pathlib import Path

import torch
import torch.distributed as dist
from torch import inf

from tensorboardX import SummaryWriter
from collections import OrderedDict
import argparse
import timm
from models.patcher import *
import warnings


def get_args_parser():
    parser = argparse.ArgumentParser('TinySaver', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, help='Per GPU batch size')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int, help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='convnextv2_base', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='image input size')

    # Optimization parameters
    parser.add_argument(
        '--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)'
    )
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-6,
        metavar='LR',
        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)'
    )
    parser.add_argument(
        '--warmup_epochs', type=float, default=20, metavar='N', help='epochs to warmup LR, if scheduler supports'
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=-1,
        metavar='N',
        help='num of steps to warmup LR, will overload warmup_epochs if set > 0'
    )

    parser.add_argument('--opt', default='AdamW', type=str, metavar='OPTIMIZER', help='Optimizer (default: "AdamW"')
    # parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    # parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0., help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0., help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument(
        '--cutmix_minmax',
        type=float,
        nargs='+',
        default=None,
        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)'
    )
    parser.add_argument(
        '--mixup_prob',
        type=float,
        default=1.0,
        help='Probability of performing mixup or cutmix when either/both is enabled'
    )
    parser.add_argument(
        '--mixup_switch_prob',
        type=float,
        default=0.5,
        help='Probability of switching to cutmix when both mixup and cutmix enabled'
    )
    parser.add_argument(
        '--mixup_mode',
        type=str,
        default='batch',
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"'
    )
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    # Dataset parameters
    parser.add_argument('--data_path', default='/imagenet', type=str, help='dataset path')
    parser.add_argument('--partial_datapath', type=str, default=None, help="secondary dataset path")
    parser.add_argument('--nb_classes', default=1000, type=int, help='number of the classification types')
    parser.add_argument('--output_dir', default='./training/', help='path where to save, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='default', help='path where to tensorboard log')
    parser.add_argument('--record_dir', default='./records/', help='path where to tensorboard log')
    # parser.add_argument('--tensorrt_dir', default='./training/tensorrt', help='path where to tensorboard log')

    parser.add_argument('--device', type=str, default=None, help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--eval_data_path', default=None, type=str, help='dataset path for evaluation')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument(
        '--data_set',
        default='IMNET',
        choices=['CIFAR', 'IMNET', 'image_folder', 'CIFAR10'],
        type=str,
        help='ImageNet dataset path'
    )
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=1, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False, help='Perform evaluation only')
    # parser.add_argument('--dist_eval', type=str2bool, default=True, help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False, help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument(
        '--pin_mem',
        type=str2bool,
        default=True,
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )

    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # distributed training parameters
    # parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    # parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument(
        '--use_amp', type=str2bool, default=False, help="Use apex AMP (Automatic Mixed Precision) or not"
    )

    parser.add_argument('--saver_model', type=str, default='convnextv2_base', help="Name of the saver model")
    parser.add_argument('--ee_dim', default=64, type=int, help='dim for early exits')
    parser.add_argument('--num_attn_layer', default=2, type=int, help='number of attention layers')
    parser.add_argument('--allowed_exits', type=str, default='None', help="List of allowed exits")

    parser.add_argument('--debug', type=str2bool, default=False, help="Debug mode")
    parser.add_argument('--force_refresh', type=str2bool, default=False, help="Force refresh the simulation")
    parser.add_argument('--sim_index', type=str, default='temp', help="Simulation index")
    parser.add_argument('--job_index', default='', type=str, help='record of job id')
    parser.add_argument('--notes', type=str, default=None, help="Notes for the experiment")
    parser.add_argument('--gpuaug', type=str2bool, default=False, help='Run augmentation on GPU or not')
    parser.add_argument('--record_pred', type=str2bool, default=True, help='Record prediction at the end of training')
    parser.add_argument(
        '--skip_training', type=str2bool, default=False, help='Run record prediction only, skip training'
    )
    # parser.add_argument('--alpha', type=float, default=0.5, help='alpha for loss function')
    # parser.add_argument('--shuffle', type=str2bool, default=True, help='Shuffle dataset')
    parser.add_argument('--dataloader', type=str, default='default', help='Which dataloader to use')
    parser.add_argument('--prefetch_factor', default=2, type=int)
    parser.add_argument('--eval_interval', default=1, type=int)

    return parser


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None, monitor=True):
        if fmt is None:
            fmt = "{median:.2f}({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
        self.monitor = monitor

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class SmoothedVector(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, fmt=None, monitor=True):
        if fmt is None:
            fmt = "{mean_reduce:.4f}"
        self.total = 0
        self.count = 0
        self.fmt = fmt
        self.monitor = monitor

    def update(self, value, n=1):
        n = torch.ones_like(value) * n
        n = torch.where(value == 0, 0, n)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        # t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(self.count)
        dist.all_reduce(self.total)
        # t = t.tolist()
        # self.count = t[0]
        # self.total = t[1]

    @property
    def avg(self):
        return self.global_avg

    @property
    def global_avg(self):
        if isinstance(self.count, torch.Tensor):
            count = torch.where(self.count == 0, 100, self.count)
            avg = self.total / count
        else:
            avg = self.total / self.count
        if isinstance(avg, torch.Tensor):
            if avg.numel() > 1:
                avg = avg.detach().cpu().numpy()
            else:
                avg = avg.item()
        return avg

    @property
    def mean_reduce(self):
        reduce = self.global_avg
        if isinstance(reduce, (float, int)):
            return reduce
        else:
            return reduce.mean()

    def __str__(self):
        return self.fmt.format(mean_reduce=self.mean_reduce)


class AveragedMatrix(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, fmt=None):
        if fmt is None:
            fmt = "{mean_reduce:.4f}"
        self.total = 0
        self.count = 0
        self.fmt = fmt

    def update(self, value):
        n = value.shape[0]
        self.count += n
        self.total += value.sum(dim=0)

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        # t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(self.count)
        dist.all_reduce(self.total)
        # t = t.tolist()
        # self.count = t[0]
        # self.total = t[1]

    @property
    def avg(self):
        return self.global_avg

    @property
    def global_avg(self):
        if isinstance(self.count, torch.Tensor):
            count = torch.where(self.count == 0, 100, self.count)
            avg = self.total / count
        else:
            avg = self.total / self.count
        if isinstance(avg, torch.Tensor):
            if avg.numel() > 1:
                avg = avg.detach().cpu().numpy()
            else:
                avg = avg.item()
        return avg

    @property
    def mean_reduce(self):
        reduce = self.global_avg
        if isinstance(reduce, (float, int)):
            return reduce
        else:
            return reduce.mean()

    def __str__(self):
        return self.fmt.format(mean_reduce=self.mean_reduce)


class MetricLogger(object):

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                if v.numel() > 1 or not isinstance(self.meters[k], SmoothedValue):
                    self.meters[k].update(v)
                    continue
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        logs_str = []
        for name, meter in self.meters.items():
            if meter.monitor:
                logs_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(logs_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def _log_meters(self):
        logs_str = []
        for name, meter in self.meters.items():
            if meter.monitor:
                logs_str.append(str(meter))
        return self.delimiter.join(logs_str)

    def _log_header(self, prefix):
        header_str = prefix
        header_str += self.delimiter
        header_str += f'{"eta":<10}{self.delimiter}'
        for name, meter in self.meters.items():
            if meter.monitor:
                header_str += ('{0:<' + str(len(str(meter))) + '}{1}').format(name, self.delimiter)
        header_str += f'{"cmpt_time":<10}{self.delimiter}'
        header_str += f'{"data_time":<10}{self.delimiter}'

        return header_str

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.3f}')
        data_time = SmoothedValue(fmt='{avg:.3f}')
        print(type(iterable))
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [header, '[{0' + space_fmt + '}/{1}]', '{eta:<10}', '{meters}', '{time:<10}', '{data:<10}']
        if torch.cuda.is_available():
            log_msg.append('{memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if i == 0:
                    prefix = ' ' * (len(header) + (2 * len(str(len(iterable)))) + 3) + self.delimiter
                    header_str = self._log_header(prefix)
                    if torch.cuda.is_available():
                        header_str += f'{"max_mem":<10}{self.delimiter}'
                    print(header_str)
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=self._log_meters(),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=self._log_meters(),
                            time=str(iter_time),
                            data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):

    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def update_hist(self, head='hist', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            self.writer.add_histogram(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


class WandbLogger(object):

    def __init__(self, args):
        self.args = args

        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        # Initialize a W&B run
        if self._wandb.run is None:
            self._wandb.init(project=args.project, config=args)

    def log_epoch_metrics(self, metrics, commit=True):
        """
        Log train/test metrics onto W&B.
        """
        # Log number of model parameters as W&B summary
        self._wandb.summary['n_parameters'] = metrics.get('n_parameters', None)
        metrics.pop('n_parameters', None)

        # Log current epoch
        self._wandb.log({'epoch': metrics.get('epoch')}, commit=False)
        metrics.pop('epoch')

        for k, v in metrics.items():
            if 'train' in k:
                self._wandb.log({f'Global Train/{k}': v}, commit=False)
            elif 'test' in k:
                self._wandb.log({f'Global Test/{k}': v}, commit=False)

        self._wandb.log({})

    def log_checkpoints(self):
        output_dir = self.args.output_dir
        model_artifact = self._wandb.Artifact(self._wandb.run.id + "_model", type="model")

        model_artifact.add_dir(output_dir)
        self._wandb.log_artifact(model_artifact, aliases=["latest", "best"])

    def set_steps(self):
        # Set global training step
        self._wandb.define_metric('Rank-0 Batch Wise/*', step_metric='Rank-0 Batch Wise/global_train_step')
        # Set epoch-wise step
        self._wandb.define_metric('Global Train/*', step_metric='epoch')
        self._wandb.define_metric('Global Test/*', step_metric='epoch')


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(f'[{datetime.datetime.now().strftime("%H:%M:%S")}]', *args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):

    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # elif 'SLURM_PROCID' in os.environ:
    #     args.rank = int(os.environ['SLURM_PROCID'])
    #     args.gpu = args.rank % torch.cuda.device_count()

    #     os.environ['RANK'] = str(args.rank)
    #     os.environ['LOCAL_RANK'] = str(args.gpu)
    #     os.environ['WORLD_SIZE'] = str(args.world_size)
    else:
        print('Not using distributed mode')
        setup_for_distributed(True)
        args.distributed = False
        return

    args.distributed = True

    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(args.rank, args.dist_url, args.gpu), flush=True)

    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(args.gpu)
    print('init finished', flush=True)
    torch.distributed.barrier()
    print('pass barrier', flush=True)
    setup_for_distributed(args.rank == 0)


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys
            )
        )
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def save_model(args, epoch, model, optimizer, loss_scaler=None, model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            # 'scaler': loss_scaler.state_dict(),
            'args': args,
        }

        if model_ema is not None:
            to_save['model_ema'] = get_state_dict(model_ema)

        save_on_master(to_save, checkpoint_path)

    if is_main_process() and isinstance(epoch, int):
        to_del = epoch - args.save_ckpt_num * args.save_ckpt_freq
        old_ckpt = output_dir / ('checkpoint-%s.pth' % to_del)
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)


def auto_load_model(args, model, optimizer, loss_scaler=None, model_ema=None):
    output_dir = Path(args.output_dir)
    if args.auto_resume and len(args.resume) == 0:
        import glob
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_var_keys = list(checkpoint['model'].keys())
        for k in model_var_keys:
            if k.startswith('module'):
                checkpoint['model'][k.removeprefix('module.')] = checkpoint['model'][k]
                del checkpoint['model'][k]
        model.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not isinstance(checkpoint['epoch'], str):  # does not support resuming with 'best', 'best-ema'
                args.start_epoch = checkpoint['epoch'] + 1
            else:
                assert args.eval, 'Does not support resuming with checkpoint-best'
            if hasattr(args, 'model_ema') and args.model_ema:
                if 'model_ema' in checkpoint.keys():
                    model_ema.module.load_state_dict(checkpoint['model_ema'])
                    print('Load EMA model')
                else:
                    model_ema.module.load_state_dict(checkpoint['model'])
            if loss_scaler is not None and 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            # print("With optim & sched!")


def load_model(args, model, ema=False):
    output_dir = Path(args.output_dir)
    import glob
    all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
    latest_ckpt = -1
    for ckpt in all_checkpoints:
        t = ckpt.split('-')[-1].split('.')[0]
        if t.isdigit():
            latest_ckpt = max(int(t), latest_ckpt)
    if latest_ckpt >= 0:
        args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
    print("Auto resume checkpoint: %s" % args.resume)
    checkpoint = torch.load(args.resume, map_location='cpu')

    if ema:
        print('Loading EMA model')
        dict_to_load = checkpoint['model_ema']
    else:
        dict_to_load = checkpoint['model']
        print('Loading non-EMA model')

    model_var_keys = list(dict_to_load.keys())
    for k in model_var_keys:
        new_key = k.replace('_orig_mod.', '')
        if k.startswith('module'):
            new_key = new_key.removeprefix('module.')
        dict_to_load[new_key] = dict_to_load[k]
        del dict_to_load[k]
    model.load_state_dict(dict_to_load)


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, warmup_steps=-1
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters]
    )

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


class CosineAnnealingWarmUp(torch.optim.lr_scheduler.LRScheduler):

    def __init__(self, optimizer, warmup_epochs, T_max, eta_min=0, last_epoch=-1, verbose=False):
        self.eta_min = eta_min
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.", UserWarning
            )

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * self.last_epoch / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            return [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * \
                (1. + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs) ))
                for base_lr in self.base_lrs
            ]



def build_model(
    model_name: str,
    device: str | int | torch.device,
    state_dict: OrderedDict = None,
    strict_dict_load: bool = True,
    freeze: bool = True,
    **model_init_args
) -> torch.nn.Module:

    model_init_args.setdefault('pretrained', state_dict is None)
    interm_feat = model_init_args.pop('interm_feat', False)
    profiling = model_init_args.pop('profiling', 0)
    no_head = model_init_args.pop('no_head', False)

    if model_name.startswith('resnet'):
        model_name += '.tv_in1k'
        patcher = ResNetPatcher
    elif model_name.startswith('convnextv2'):
        model_name += '.fcmae_ft_in1k'
        patcher = ConvNextPatcher
    elif model_name.startswith('davit'):
        # ['davit_base', 'davit_giant', 'davit_huge', 'davit_large', 'davit_small', 'davit_tiny']
        if model_name.startswith('davit_tiny'):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        patcher = DaViTPatcher
    elif model_name.startswith('swin'):
        # ['swin_base_patch4_window7_224.ms_in22k_ft_in1k', 'swin_large_patch4_window7_224.ms_in22k_ft_in1k', 'swin_small_patch4_window7_224.ms_in22k_ft_in1k', 'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k']
        model_name += '_patch4_window7_224.ms_in22k_ft_in1k'
        patcher = SwinTPatcher

    elif model_name.startswith('maxvit'):
        # ['maxvit_base_tf_224.in1k', 'maxvit_large_tf_224.in1k', 'maxvit_small_tf_224.in1k',  'maxvit_tiny_tf_224.in1k']
        model_name += '_tf_224.in1k'
        patcher = MaxVitPatcher
    elif model_name.startswith('mobilenetv3'):
        # ['maxvit_base_tf_224.in1k', 'maxvit_large_tf_224.in1k', 'maxvit_small_tf_224.in1k',  'maxvit_tiny_tf_224.in1k']
        if model_name == 'mobilenetv3_large_100':
            model_name = 'mobilenetv3_large_100.ra_in1k'
        elif model_name == 'mobilenetv3_small_100':
            model_name = 'mobilenetv3_small_100.lamb_in1k'
        else:
            raise NotImplementedError
        patcher = MobileNetv3Patcher

    elif model_name.startswith('efficientnet'):
        # efficientnet_b0-b4
        if model_name == 'efficientnet_b0':
            model_name = 'efficientnet_b0.ra_in1k'
        elif model_name == 'efficientnet_b1':
            model_name = 'efficientnet_b1.ft_in1k'
        elif model_name == 'efficientnet_b2':
            model_name = 'efficientnet_b2.ra_in1k'
        elif model_name == 'efficientnet_b3':
            model_name = 'efficientnet_b3.ra2_in1k'
        elif model_name == 'efficientnet_b4':
            model_name = 'efficientnet_b4.ra2_in1k'
        else:
            raise NotImplementedError
        patcher = EfficientNetPatcher
    elif model_name.startswith('efficientvit'):
        # m0-m5,b0-b3
        model_name += '.r224_in1k'
        patcher = EfficientViTPatcher
    elif model_name.startswith('efficientformerv2'):
        # s0-2,l
        model_name += '.snap_dist_in1k'
        patcher = EfficientFormerv2Patcher
    else:
        raise NotImplementedError
    model = timm.create_model(f'{model_name}', **model_init_args).to(device)
    if interm_feat or profiling > 0 or no_head:
        model = patcher().patch(model, no_head, profiling)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=strict_dict_load)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model


model_abbr_name = {
    'cn2l': 'convnextv2_large',
    'cn2h': 'convnextv2_huge',
    'cn2b': 'convnextv2_base',
    'cn2n': 'convnextv2_nano',
    'cn2p': 'convnextv2_pico',
    'cn2f': 'convnextv2_femto',
    'cn2a': 'convnextv2_atto',
    'cn2t': 'convnextv2_tiny',
    'r152': 'resnet152',
    'r101': 'resnet101',
    'r50': 'resnet50',
    'r34': 'resnet34',
    'r18': 'resnet18',
    'swinb': 'swin_base',
    'swinl': 'swin_large',
    'swint': 'swin_tiny',
    'swins': 'swin_small',
    'mnetv3l100': 'mobilenetv3_large_100',
    'mnetv3s100': 'mobilenetv3_small_100',
    'enetb0': 'efficientnet_b0',
    'enetb1': 'efficientnet_b1',
    'enetb2': 'efficientnet_b2',
    'enetb3': 'efficientnet_b3',
    'enetb4': 'efficientnet_b4',
    'davitb': 'davit_base',
    'davits': 'davit_small',
    'davitt': 'davit_tiny',
    'maxvitb': 'maxvit_base',
    'maxvitl': 'maxvit_large',
    'maxvits': 'maxvit_small',
    'maxvitt': 'maxvit_tiny',
    'evitb0': 'efficientvit_b0',
    'evitb1': 'efficientvit_b1',
    'evitb2': 'efficientvit_b2',
    'evitb3': 'efficientvit_b3',
    'evitm0': 'efficientvit_m0',
    'evitm1': 'efficientvit_m1',
    'evitm2': 'efficientvit_m2',
    'evitm3': 'efficientvit_m3',
    'evitm4': 'efficientvit_m4',
    'evitm5': 'efficientvit_m5',
    'eformers0': 'efficientformerv2_s0',
    'eformers1': 'efficientformerv2_s1',
    'eformers2': 'efficientformerv2_s2',
    'eformerl': 'efficientformerv2_l',
}
model_name_abbr = {v: k for k, v in model_abbr_name.items()}
