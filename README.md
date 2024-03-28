# Tiny Models are the Computational Saver for Large Models

Code release of [https://arxiv.org/abs/2403.17726]()

## Installation

Our code requires the environment of `timm==0.9.12`, to build the env from scratch:

```bash
git submodule update --init --recursive --remote
conda create -n tinysaver python==3.10 -y
conda activate tinysaver
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Running

### Find best saver & run evaluation

Set path to imagenet

```
PATH_TO_IMAGENET=/ssd_storage/public/imagenet
```

Run

```
python exit_estimation.py --data_path $PATH_TO_IMAGENET --dataset_split train
python exit_estimation.py --data_path $PATH_TO_IMAGENET --dataset_split test
python evaluation.py --mode single
```

Then the infomation of which saver is the best is stored in `TinySaver/records/best_saver_info_train.csv`. The complexity/accuracy info for each base/saver pair is store in `TinySaver/eval/sim_results/$BASE_IMNET1k_$SAVER_single_test.csv`

### Training ESN

Edit `TinySaver/task_configs/grid_search.yaml` for `data_path` and each expriment entry

Run

```
python run_task_parallel.py --config task_configs/grid_search.yaml --tasks_in_parallel 1 --gpu_per_task 1
python evaluation.py --mode dyce
```

`--tasks_in_parallel` refers how many training to run in parallel
`--gpu_per_task` refers how many gpus to use for every training
If your machine is experiencing out-of-memory error, please consider decrease `batch_size` and increase `update_freq` at the same rate.
Results is sotred in `TinySaver/eval/sim_results/`
