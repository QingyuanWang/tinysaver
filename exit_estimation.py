import torch
import numpy as np
from utils import str2bool, build_model
from datasets import build_dataset
import argparse
import os
from engine import record_prediction
from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
import json
from macs_analysis import get_complexity
import logging
import math
from tqdm import tqdm
import pandas as pd
from utils import get_args_parser


def get_standalone_record(model_name, args, split='train'):
    if os.path.exists(f'{args.record_dir}/{model_name}_{split}.npz'):
        print(f'Loading from record: {args.record_dir}/{model_name}_{split}.npz')
        record = np.load(f'{args.record_dir}/{model_name}_{split}.npz')
    else:
        model = build_model(model_name, args.device, interm_feat=False)
        print(f'Record not found, generating: {args.record_dir}/{model_name}_{split}.npz')
        model = torch.compile(model)
        print(f'Batch size: {args.batch_size}')
        if args.batch_size > 1024:
            args.gpuaug = True
        else:
            args.gpuaug = False
        if hasattr(args, 'dataset_train'):
            dataset_train = args.dataset_train
            transform_gpu = args.transform_gpu
        else:
            args.batch_size = 1024
            dataset_train, args.nb_classes, transform_gpu = build_dataset(
                is_train=split == 'train', eval_mode=False, args=args
            )
            args.dataset_train = dataset_train
            args.transform_gpu = transform_gpu
        sampler_train = torch.utils.data.SequentialSampler(dataset_train)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            # pin_memory=args.pin_mem,
            # prefetch_factor=4,
            drop_last=False,
        )

        record = record_prediction(
            data_loader_train,
            model,
            args.device,
            f'{args.record_dir}/{model_name}_{split}.npz',
            transform_gpu=transform_gpu
        )
    confs = record['confs']
    preds = record['preds']
    label = record['label'].squeeze()
    return confs, preds, label


def get_complexity_info(model_name, args):
    if os.path.exists(f'{args.record_dir}/complexity_info.json'):
        with open(f'{args.record_dir}/complexity_info.json', 'r') as f:
            complexity_info = json.load(f)
        if model_name in complexity_info:
            return complexity_info[model_name]
    else:
        complexity_info = {}
    model = build_model(model_name, args.device, interm_feat=False).to(args.device)
    inp = torch.rand((1, 3, 224, 224)).to(args.device)
    model.eval()
    prof = FlopsProfiler(model)
    prof.start_profile()

    model(inp)
    flops = prof.get_total_flops(as_string=False)
    macs = prof.get_total_macs(as_string=False)
    params = prof.get_total_params(as_string=False)
    prof.end_profile()
    confs, preds, label = get_standalone_record(model_name, args, split='test')
    acc = (preds[0, :, 0] == label).sum() / label.shape[0]
    complexity_info[model_name] = {'flops': flops, 'macs': macs, 'params': params, 'acc': acc}
    with open(f'{args.record_dir}/complexity_info.json', 'w') as f:
        json.dump(complexity_info, f, indent=4)
    return complexity_info[model_name]


def delat_c(i, ee_ratio, remaining_ratio, macs_info):
    '''
    Compute DeltaC for employing i-th exit.
    i: index of the exit
    ee_ratio: early-exit ratio when the systematic perfermance is equal to the target performance
    macs_info: macs info of the model
    remaining_ratio: ratio of remaining samples (will exited by the final exit)
    '''
    prior_backbone_cost = macs_info['backbone']['macs'][i] / macs_info['backbone']['ori_macs']
    if i == 0:
        exit_cost = (macs_info['attn']['macs'][i] + macs_info['head'][0]['macs'][i]) / macs_info['backbone']['ori_macs']
    else:
        exit_cost = (macs_info['attn']['macs'][i] - macs_info['attn']['macs'][i - 1] +
                     macs_info['head'][0]['macs'][i]) / macs_info['backbone']['ori_macs']

    # print('deltac', prior_backbone_cost, exit_cost, ee_ratio, remaining_ratio)
    return ee_ratio * (prior_backbone_cost + exit_cost - 1) + remaining_ratio * exit_cost


def match_performance(i, ee_confs, ee_preds, tgt_preds, label, macs_info, num_samples, compelete=False):
    info = []
    matched_info = None
    for t in tqdm(np.arange(1, 0, -0.001)):
        ee_mask = (ee_confs > t)
        pass_mask = (ee_confs <= t)
        ee_preds_sel = ee_preds[ee_mask]
        tgt_preds_sel = tgt_preds[ee_mask]
        tgt_preds_pass = tgt_preds[pass_mask]
        label_sel = label[ee_mask]
        label_pass = label[pass_mask]

        # consistent = (ee_preds_sel == tgt_preds_sel).sum() / ee_preds_sel.shape[0]
        if ee_preds_sel.shape[0] > 0:
            ee_acc = (ee_preds_sel == label_sel).sum() / ee_preds_sel.shape[0]
            tgt_acc = (tgt_preds_sel == label_sel).sum() / tgt_preds_sel.shape[0]
        else:
            ee_acc = 0
            tgt_acc = 0
        ee_ratio = ee_preds_sel.shape[0] / num_samples
        pass_ratio = tgt_preds_pass.shape[0] / num_samples

        acc = ((ee_preds_sel == label_sel).sum() + (tgt_preds_pass == label_pass).sum()) / label.shape[0]
        # print(f'Threshold: {t}, Overall acc: {acc}, EE acc: {ee_acc}, TGT acc: {tgt_acc}, EE rate: {ee_ratio}')
        info.append(
            {
                'threshold': t,
                'overall_acc': acc,
                'ee_acc': ee_acc,
                'tgt_acc': tgt_acc,
                'ee_ratio': ee_ratio,
                'pass_ratio': pass_ratio,
                'DeltaC': delat_c(i, ee_ratio, pass_ratio, macs_info),
            }
        )
        # print(info[-1])
        if ee_acc < tgt_acc and matched_info is None:
            matched_info = info[-1]
            if not compelete:
                break
    if matched_info is None:
        matched_info = info[-1]
    if compelete:
        df = pd.DataFrame(info)
        df.to_csv(
            f'{args.record_dir}/{args.model}_{args.saver_model}_{args.dataset_split}_exit_estimation.csv', index=False
        )
        # print(ee_preds_sel.shape[0], sum(ee_mask), sum(pass_mask), ee_ratio)
        print(f'Saved {args.record_dir}/{args.model}_{args.saver_model}_{args.dataset_split}_exit_estimation.csv')
    return matched_info, pass_mask


def get_record(args):
    confs, preds, label = get_standalone_record(args.saver_model, args, split=args.dataset_split)
    ee_confs = confs[:, :, 0]
    ee_preds = preds[:, :, 0]

    confs, preds, label = get_standalone_record(args.model, args, split=args.dataset_split)
    tgt_preds = preds[0, :, 0]
    tgt_acc = (tgt_preds == label).sum() / label.shape[0]

    return ee_confs, ee_preds, tgt_preds, tgt_acc, label, label.shape[0]


def main(args):

    if os.path.exists(f'{args.record_dir}/exit_estimation_{args.dataset_split}.json'):
        with open(f'{args.record_dir}/exit_estimation_{args.dataset_split}.json', 'r') as f:
            exit_estimation = json.load(f)
        if args.model in exit_estimation and args.saver_model in exit_estimation[args.model]:
            print(
                f'Exit estimation of {args.model} and {args.saver_model} exists: {exit_estimation[args.model][args.saver_model]}'
            )
            return
    else:
        exit_estimation = {}
    if args.model not in exit_estimation:
        exit_estimation[args.model] = {}
    if args.saver_model not in exit_estimation[args.model]:
        exit_estimation[args.model][args.saver_model] = {}
    base_info = get_complexity_info(args.model, args)
    saver_info = get_complexity_info(args.saver_model, args)
    if saver_info['macs'] >= base_info['macs']:
        print(f'Error: {args.saver_model} does not have fewer MACs than {args.model}, Skip.')
        return
    macs_dict = {
        'backbone': {
            'macs': [0, base_info['macs']],
            'ori_macs': base_info['macs'],
            'ori_params': base_info['params'],
        },
        'head': [{
            'macs': [0, 0]
        }],
        'attn': {
            'macs': [saver_info['macs'], saver_info['macs']]
        }
    }
    # print(macs_dict)

    assert len(macs_dict['backbone']['macs']) == len(macs_dict['head'][0]['macs'])
    assert len(macs_dict['backbone']['macs']) == len(macs_dict['attn']['macs'])
    args.batch_size = 2**int(math.log(10**11 // (macs_dict['backbone']['ori_params']), 2))
    for _ in range(10):
        try:
            ee_confs, ee_preds, tgt_preds, tgt_acc, label, num_samples = get_record(args)
            break
        except Exception as e:
            print(e)
            args.batch_size = int(args.batch_size / 2)
            print(f'Retry with batch size: {args.batch_size}')

    sample_arrived_mask = np.ones((ee_confs.shape[1]), dtype=bool)

    # print(ee_confs.shape, ee_preds.shape, tgt_preds.shape, tgt_acc, label.shape, num_samples)
    for i in range(ee_confs.shape[0]):
        confs_arrived = ee_confs[i, sample_arrived_mask]
        preds_arrived = ee_preds[i, sample_arrived_mask]
        tgt_preds_arrived = tgt_preds[sample_arrived_mask]
        label_arrived = label[sample_arrived_mask]
        # print(label_arrived.shape, sum(sample_arrived_mask))
        matched_info, remaining_mask = match_performance(
            i,
            confs_arrived,
            preds_arrived,
            tgt_preds_arrived,
            label_arrived,
            macs_dict,
            num_samples,
            compelete=args.save_full
        )
        exit_estimation[args.model][args.saver_model][i] = matched_info
        if i == 0 or matched_info['DeltaC'] < 0:
            sample_arrived_mask[sample_arrived_mask] = remaining_mask
        exit_estimation[args.model][args.saver_model][i]['enabled'] = matched_info['DeltaC'] < 0
        if sample_arrived_mask.sum() == 0:
            break
    # print(tgt_acc)

    with open(f'{args.record_dir}/exit_estimation_{args.dataset_split}.json', 'w') as f:
        json.dump(exit_estimation, f, indent=4)


if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser('TinySaver Simulation', parents=[get_args_parser()])
    parser.add_argument(
        '--save_full', type=str2bool, default=True, help='Do not random erase first (clean) augmentation split'
    )
    parser.add_argument(
        '--dataset_split', type=str, default='test', help='Do not random erase first (clean) augmentation split'
    )
    args = parser.parse_args()
    model_list = [
 "convnextv2_atto",
 "resnet18", "swin_small", 
    ]

    if not os.path.exists(args.record_dir):
        os.makedirs(args.record_dir)

    for base_model in model_list:
        for saver_model in model_list:
            print(f'Proscessing Base model: {base_model} and Saver model: {saver_model}')
            args.model = base_model
            args.saver_model = saver_model
            main(args)

    estimation = json.load(open(f'{args.record_dir}/exit_estimation_{args.dataset_split}.json'))
    compinfo = json.load(open(f'{args.record_dir}/complexity_info.json'))

    best_saver_info = []
    for base, savers in estimation.items():
        print(base)
        minDeltaC = 100
        bestSaver = None
        # bestSaverInfo = None
        saver_info = [
            (
                saver,
                info['0']['DeltaC'],
                info['0']['ee_ratio'],
                1 - (compinfo[base]['acc'] - compinfo[saver]['acc']),
            ) for saver, info in savers.items()
        ]
        saver_info.sort(key=lambda x: x[1])
        bestSaver, minDeltaC, ee_ratio, ratio_ub = saver_info[0]
        print(f'The best saver among {len(savers)} candidate for {base} is {bestSaver} DeltaC = {minDeltaC}')
        for saver, deltaC, ee_ratio, ratio_ub in saver_info[:5]:
            print(
                f'{saver} \t\twith DeltaC = {deltaC} test: {estimation[base][saver]["0"]["DeltaC"]} {estimation[base][saver]["0"]["ee_ratio"]}'
            )
        print('-' * 40)
        best_saver_info.append(
            {
                'base_model': base,
                'best_saver': bestSaver,
                'DeltaC': minDeltaC,
                'candidate_savers':
                    {saver: (deltaC, ee_ratio, ratio_ub) for saver, deltaC, ee_ratio, ratio_ub in saver_info[:5]}
            }
        )

    pd.DataFrame(best_saver_info).to_csv(f'{args.record_dir}/best_saver_info_{args.dataset_split}.csv', index=False)
