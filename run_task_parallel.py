import os
import multiprocessing as mp
import traceback
import argparse
import yaml
import datetime
import subprocess
import threading
import concurrent.futures
import re
import random


def run_mp(task_cmd):
    name = threading.current_thread().name
    rank = int(name.split('_')[1])
    time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    port = 36587 + rank
    gpu_alloc = f'CUDA_VISIBLE_DEVICES={args.gpu_per_task*rank}'
    for i in range(args.gpu_per_task - 1):
        gpu_alloc += f',{args.gpu_per_task*rank+i+1}'
    sim_index = re.search(r'--sim_index\s+(\S+)', task_cmd).group(1)
    try:
        fp = open(f'./running_log/{time}-{sim_index}_{rank}.log', "a")
    except:
        print('Error while open fp', f'./running_log/{time}-{sim_index}_{rank}.log')
        traceback.print_exc()
    task_cmd = task_cmd.replace('PORT', str(port))
    try:
        print(f'{gpu_alloc} {task_cmd}')
        proc = subprocess.Popen(f'{gpu_alloc} {task_cmd}', shell=True, stdout=fp, stderr=fp)
        proc.wait()
    except:
        print('Error for', sim_index)
        traceback.print_exc()
    finally:
        fp.close()
    # print(sim_index, f'cuda:{rank}')


def override_args(task_args):

    return task_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks_in_parallel', type=int, default=1, help='Number of task to run in parallel')
    parser.add_argument('--gpu_per_task', type=int, default=2, help='Number of gpu per task')
    parser.add_argument('--config', type=str, default='./task_config.yaml')
    parser.add_argument('--override', type=str, default='{}')
    args = parser.parse_args()

    os.makedirs('./running_log', exist_ok=True)
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    tasks = []
    global_settings = config['global']
    global_settings['params'].update(eval(args.override))

    print(f'Running {len(config["tasks"])} tasks')
    for task in config['tasks']:
        settings = global_settings.copy()
        if 'exec' in task:
            settings['exec'] = task['exec']
        settings['params'].update(task['params'])
        cmd = ''
        cmd += settings["exec"].replace('NUM_PROC', str(args.gpu_per_task))
        for k, v in settings['params'].items():
            cmd += f' \\\n --{k} {v}'
        ep = settings['params']['epochs']
        sim_index = settings['params']['sim_index']
        output_dir = settings['params']['output_dir']
        record_dir = settings['params']['record_dir']
        if os.path.exists(f'{record_dir}/{sim_index}/{sim_index}_train.npz') and os.path.exists(
            f'{record_dir}/{sim_index}/{sim_index}_test.npz'
        ) and not settings["params"]["force_refresh"]:
            print(f'./records/{sim_index}/{sim_index} exists, skip.')
            continue
        if os.path.exists(f'./{output_dir}/{sim_index}/checkpoint-{ep-1}.pth'
                         ) and 'record_pred' in settings['params'] and not settings['params']['record_pred']:
            print(f'./{output_dir}/{sim_index} training finished, skip.')
            continue
        if not os.path.exists(f'./{output_dir}/{sim_index}/checkpoint-{ep-1}.pth'
                             ) and 'skip_training' in settings['params'] and settings['params']['skip_training']:
            print(f'./{output_dir}/{sim_index} training not finished, cannot do inference.')
            continue
        print(F'Running {sim_index} with {ep} epochs')
        tasks.append(cmd)

    print(len(tasks))
    random.shuffle(tasks)
    mp.set_start_method('spawn')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.tasks_in_parallel) as executor:
        executor.map(run_mp, tasks)

        executor.shutdown()
