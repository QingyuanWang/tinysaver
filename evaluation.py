import os
from config_search import dyce_analysis, single_analysis
import multiprocessing as mp
import torch
import traceback
import sys
import argparse
from utils import model_abbr_name, model_name_abbr
import datetime
from utils import str2bool
import pandas as pd


def dyce_evaluation(sim_index, args, refresh=False, device='cuda'):
    siminfo = sim_index.split('_')
    base_model_abbr = siminfo[0]
    saver_model_abbr = siminfo[5]
    base_model_name = model_abbr_name[base_model_abbr]
    saver_model_name = model_abbr_name[saver_model_abbr]
    dim = int(siminfo[4].replace('d', ''))
    num_attn_layer = int(siminfo[3].replace('attn', ''))
    if not os.path.exists(f'{args.record_path}/{sim_index}'):
        print(f'{args.record_path}/{sim_index} not exists, abort.')
        return
    print(f'Evaluation for {args.record_path}/{sim_index}')

    dyce_analysis(
        sim_index=sim_index,
        base_model_name=base_model_name,
        saver_model_name=saver_model_name,
        dim=dim,
        search_step=0.001,
        allowed_exits=None,
        search_range=[0.5, 0.51],
        refresh=refresh,
        device=device,
        num_attn_layer=num_attn_layer,
        record_dir=args.record_path,
        save_dir=f'{args.eval_path}/sim_results',
    )


def single_evaluation(sim_index, args, refresh=False, device='cuda'):
    siminfo = sim_index.split('_')
    base_model_abbr = siminfo[0]
    saver_model_abbr = siminfo[2]
    base_model_name = model_abbr_name[base_model_abbr]
    saver_model_name = model_abbr_name[saver_model_abbr]

    single_analysis(
        sim_index=sim_index,
        base_model_name=base_model_name,
        saver_model_name=saver_model_name,
        search_step=0.001,
        search_range=[0, 1.001],
        refresh=refresh,
        device=device,
        record_dir=args.record_path,
        save_dir=f'{args.eval_path}/sim_results',
        dataset_split=args.dataset_split
    )


def evaluation_mp(arg):
    sim_index, args = arg
    rank = mp.current_process()._identity[0] - 1
    time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    fp = open(f'{args.eval_path}/sim_results/log/{time}-{sim_index}_{rank}.log', "a")
    sys.stdout = fp
    sys.stderr = fp
    device = 'cpu' if args.cpu else f'cuda:{rank}'
    print(sim_index, device)
    try:
        if args.mode == 'dyce':
            dyce_evaluation(sim_index, args, device=device)
        elif args.mode == 'single':
            single_evaluation(sim_index, args, device=device)
    except:
        print('Error for', sim_index)
        traceback.print_exc()
    finally:
        fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_path', type=str, default='./records')
    parser.add_argument('--eval_path', type=str, default='./eval')
    parser.add_argument('--mode', type=str, choices=['dyce', 'single'], default='dyce')
    parser.add_argument('--cpu', type=str2bool, default=False)
    parser.add_argument('--nproc', type=int, default=-1)
    parser.add_argument('--dataset_split', type=str, default='test')
    args = parser.parse_args()
    sims_ = os.listdir(args.record_path)

    os.makedirs(f'{args.eval_path}/sim_results/log', exist_ok=True)
    sims = []
    if args.mode == 'dyce':
        for sim in sims_:
            if '.' in sim:
                continue
            if os.path.exists(f'{args.eval_path}/sim_results/{sim}_onepass.csv'):
                print(f'{args.eval_path}/sim_results/{sim}_onepass.csv exists, skip.')
            elif not os.path.exists(f'{args.record_path}/{sim}/{sim}_train.npz'):
                print(f'Record {args.record_path}/{sim}/{sim}_train.npz not exists, abort.')
            elif not os.path.exists(f'{args.record_path}/{sim}/{sim}_test.npz'):
                print(f'Record {args.record_path}/{sim}/{sim}_test.npz not exists, abort.')
            else:
                print(f'Evaluation for {args.record_path}/{sim}')
                sims.append((sim, args))
    elif args.mode == 'single':
        if not os.path.exists(f'{args.record_path}/best_saver_info_train.csv'):
            print(
                f'{args.record_path}/best_saver_info_train.csv not exists, please run exit_estimation.py --dataset_split train first.'
            )
            exit(0)
        best_saver_info = pd.read_csv(f'{args.record_path}/best_saver_info_train.csv')
        for i, row in best_saver_info.iterrows():
            for saver_abbr in eval(row['candidate_savers']).keys():
                sim = f'{model_name_abbr[row["base_model"]]}_IMNET1k_{model_name_abbr[saver_abbr]}'
                # sim = f'{model_name_abbr[row["base_model"]]}_IMNET1k_{model_name_abbr[row["best_saver"]]}'
                if os.path.exists(f'{args.eval_path}/sim_results/{sim}_single_{args.dataset_split}.csv'):
                    print(f'{args.eval_path}/sim_results/{sim}_single_{args.dataset_split}.csv exists, skip.')
                    break
                if not os.path.exists(f'{args.record_path}/{row["base_model"]}_test.npz'):
                    print(f'Record {args.record_path}/{row["base_model"]}_test.npz not exists, abort.')
                    break
                if not os.path.exists(f'{args.record_path}/{row["best_saver"]}_test.npz'):
                    print(f'Record {args.record_path}/{row["best_saver"]}_test.npz not exists, abort.')
                    break
                print(f'Evaluation for single ee: {sim}')
                sims.append((sim, args))

    mp.set_start_method('spawn')
    if args.nproc == -1:
        args.nproc = torch.cuda.device_count()
    with mp.Pool(args.nproc) as p:
        p.map(evaluation_mp, sims)
