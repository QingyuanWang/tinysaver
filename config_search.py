import numpy as np
import pandas as pd
import torch

import sys
from dyce import ConfigEvalAccelerated, load_records, save_search_result
from tqdm import tqdm
from scipy.optimize import fminbound
import json
import scipy
import time
from macs_analysis import get_complexity
import os
import math


def compose_records(path, split, saver_name, base_name, macs_dict):
    record_dict = np.load(f'{path}/{saver_name}_{split}.npz', allow_pickle=True)
    label = record_dict['label']
    saver_preds = record_dict['preds'].astype(np.int16)
    saver_confs = record_dict['confs']

    record_dict = np.load(f'{path}/{base_name}_{split}.npz', allow_pickle=True)
    base_preds = record_dict['preds'].astype(np.int16)
    base_confs = record_dict['confs']
    saver_macs = macs_dict[saver_name]['macs']
    base_macs = macs_dict[base_name]['macs']
    bb_macs = [saver_macs, saver_macs + base_macs]
    exit_macs = [[0]]
    len_type = 1
    len_pos = 2
    print(len_pos, len_type)
    print(saver_confs.shape, saver_preds.shape, label.shape, len(bb_macs))
    num_samples = label.shape[0]

    sd_info = []
    base_acc = (base_preds[-1][:, 0] == label.squeeze()).sum() / num_samples
    preds_reorganized = [saver_preds, base_preds]
    confs_reorganized = [saver_confs, base_confs]

    # return (
    #     label, preds_reorganized, confs_reorganized, None, bb_macs, exit_macs, base_acc, base_macs, len_pos, len_type,
    #     num_samples, None
    # )
    return {
        'label': label,
        'preds': preds_reorganized,
        'confs': confs_reorganized,
        'bb_macs': bb_macs,
        'exit_macs': exit_macs,
        'base_acc': base_acc,
        'base_macs': base_macs,
        'len_pos': len_pos,
        'len_type': len_type,
        'num_samples': num_samples,
        'sd_info': sd_info
    }



def print_info(base_acc, base_macs, sd_info, sim_index=None, split='train'):
    if sim_index is not None:
        print(f'[{sim_index} {split}] ')
    print('\t Acc\t Acc%\t MACs\t\t MACs%')
    for i in range(len(sd_info)):
        print(f'Pos {i}')
        for j in range(len(sd_info[i])):
            print(f'\t{sd_info[i][j][0]:.4f}\t{sd_info[i][j][1]:.4f}\t{sd_info[i][j][2]:<10}\t{sd_info[i][j][3]:.4f}')
    print(f'Base acc: {base_acc}, base macs: {base_macs}')


def save_info(sd_info, save_path):
    info = []
    for i in range(len(sd_info)):
        for j in range(len(sd_info[i])):
            info.append(sd_info[i][j])
    info = pd.DataFrame(info, columns=['acc', 'acc%', 'macs', 'macs%'])
    info.to_csv(save_path, index=False)


def acc_vs_thres(label, preds_reorganized, confs_reorganized, save_path):
    preds = preds_reorganized[0][4, :, 0]
    confs = confs_reorganized[0][4, :, 0]
    info = []
    for i in np.arange(0.0, 1.01, 0.01):
        acc = ((confs > i) & (preds == label.squeeze())).sum() / (confs > i).sum()
        exiting_rate = (confs > i).sum() / label.shape[0]
        info.append([i, acc, exiting_rate])
    info = pd.DataFrame(info, columns=['thres', 'acc', 'exiting_rate'])
    info.to_csv(save_path, index=False)



def dyce_analysis(
    sim_index: str,
    base_model_name: str,
    saver_model_name: str,
    search_range: list[float] = [0.9, 1.01],
    search_step: float = 0.01,
    dim: int = 512,
    num_attn_layer=2,
    allowed_exits: None | list[int] = None,
    record_dir: str = './records/',
    save_dir: str = './eval/sim_results/',
    save_name: str = None,
    method: str = 'onepass',
    calibration: bool = False,
    device: str = 'cuda',
    refresh=False
):
    if save_name is None:
        save_name = f'{sim_index}_{method}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if not refresh and os.path.exists(f'{save_dir}/{save_name}.csv'):
        print(f'{save_dir}/{save_name}.csv exists, skip')
        return
    train_record_path = f'{record_dir}/{sim_index}/{sim_index}_train.npz'
    test_record_path = f'{record_dir}/{sim_index}/{sim_index}_test.npz'
    print(f'[{sim_index}] Start')
    print(f'[{sim_index}] Load train record from: {train_record_path}')
    macs_info = get_complexity(
        base_model_name=base_model_name,
        saver_model_name=saver_model_name,
        dim=dim,
        num_attn_layer=num_attn_layer,
        allowed_exits=allowed_exits,
        refresh=refresh,
        device=device
    )
    # quit()
    record_info = load_records(train_record_path, macs_info)
    
    print_info(record_info['bb_macs'], record_info['base_macs'], record_info['sd_info'], sim_index=sim_index, split='train')
    cea = ConfigEvalAccelerated(
        record_info['label'], record_info['preds'], record_info['confs'], record_info['exit_macs'], record_info['bb_macs'], record_info['base_acc'], record_info['base_macs'], device=device
    )
    start_time = time.time()
    configs = []
    for lam in tqdm(np.arange(*search_range, search_step)):
        configs.append({'exit_config': cea.config_search(method, lam=lam, calibration=calibration, normalize=True),'lambda': lam})
    end_time = time.time()
    print(f'[{sim_index}] Conf generated')
    print(f'[{sim_index}] Load test record from: {test_record_path}')

    record_info = load_records(test_record_path, macs_info)
    print_info(record_info['base_acc'], record_info['base_macs'], record_info['sd_info'], sim_index=sim_index, split='test')

    cea = ConfigEvalAccelerated(
        record_info['label'], record_info['preds'], record_info['confs'], record_info['exit_macs'], record_info['bb_macs'], record_info['base_acc'], record_info['base_macs'], device=device
    )
    save_search_result(cea, configs, record_info['base_macs'], record_info['base_acc'], save_dir, save_name)
    
    print(f'[{sim_index}] Time cost: {end_time - start_time}')


def single_analysis(
    sim_index: str,
    base_model_name: str,
    saver_model_name: str,
    search_range: list[float] = [0.0, 1.01],
    search_step: float = 0.01,
    record_dir: str = './exit_estimation',
    save_dir: str = './eval/sim_results/',
    save_name: str = None,
    calibration: bool = False,
    refresh=False,
    device: str = 'cuda',
    dataset_split='test'
):
    if save_name is None:
        save_name = f'{sim_index}_single_{dataset_split}'
    if not refresh and os.path.exists(f'./eval/{save_name}.csv'):
        print(f'{save_name} exists, skip')
        return
    test_record_path = f'{record_dir}/{sim_index}/{sim_index}_{dataset_split}.npz'
    print(f'Start')
    print(f'Load train record from: {test_record_path}')
    macs_info = json.load(open(f'{record_dir}/complexity_info.json', 'r'))
    print(f'Conf generated')
    print(f'Load test record from: {test_record_path}')

    record_info = compose_records(record_dir, 'test', saver_model_name, base_model_name, macs_info)
    print_info(record_info['base_acc'], record_info['base_macs'], record_info['sd_info'], sim_index=sim_index, split='test')

    cea = ConfigEvalAccelerated(
        record_info['label'], record_info['preds'], record_info['confs'], record_info['exit_macs'], record_info['bb_macs'], record_info['base_acc'], record_info['base_macs'], device=device
    )
    configs = []
    for lam in tqdm(np.arange(*search_range, search_step)):
        configs.append({'exit_config': cea.config_search('samethres', lam=lam, calibration=calibration, normalize=True),'lambda': lam})
    save_search_result(cea, configs, record_info['base_macs'], record_info['base_acc'], save_dir, save_name)