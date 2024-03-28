import torch
from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
from models.dmc import DMC_train
import json
from utils import str2bool, build_model
from functools import partial
import os
from tqdm import tqdm
import logging


@torch.no_grad()
def get_complexity(
    base_model_name,
    saver_model_name,
    input_size=(3, 224, 224),
    dim=512,
    allowed_exits=None,
    num_attn_layer=2,
    save_path=None,
    refresh=False,
    device='cuda',
):
    logging.disable(logging.INFO)
    dmc_class = DMC_train
    dmc_params = {
        'num_classes': 1000,
        'input_size': input_size,
        'dim': dim,
        'allowed_exits': allowed_exits,
        'num_attn_layer': num_attn_layer
    }
    if save_path is None:
        os.makedirs('./eval/macs_info', exist_ok=True)
        save_path = f'./eval/macs_info/dmcv3_{base_model_name}_{saver_model_name}_attn{dmc_params["num_attn_layer"]}_{dmc_params["dim"]}.json'
    if not refresh and os.path.exists(save_path):
        print(f'Exit estimation of {base_model_name} and {saver_model_name} exists: {save_path}')
        with open(save_path, 'r') as f:
            macs_dict = json.load(f)
        return macs_dict
    print('Macs info not exists, generating...')
    inp = torch.rand((1, *input_size)).to(device)
    create_model = partial(build_model, base_model_name, device, strict_dict_load=False)
    create_saver_model = partial(build_model, saver_model_name, device, strict_dict_load=False)
    model = create_model(interm_feat=True).to(device)
    model.eval()
    prof = FlopsProfiler(model)
    prof.start_profile()

    _, interm_feat = model(inp)
    if 'allowed_exits' in dmc_params and dmc_params['allowed_exits'] is not None:
        for i in range(len(dmc_params['allowed_exits'])):
            if dmc_params['allowed_exits'][i] < 0:
                dmc_params['allowed_exits'][i] += len(interm_feat)

    flops_ori = prof.get_total_flops(as_string=False)
    macs_ori = prof.get_total_macs(as_string=False)
    params_ori = prof.get_total_params(as_string=False)
    # prof.print_model_profile()
    prof.end_profile()
    print(f'GMacs: {macs_ori/(10**9):.2f}, GParams: {params_ori/(10**9):.2f}')

    model_saver = create_saver_model(interm_feat=True).to(device)
    model_saver.eval()
    prof = FlopsProfiler(model_saver)
    prof.start_profile()

    _, _ = model_saver(inp)

    flops_saver_ori = prof.get_total_flops(as_string=False)
    macs_saver_ori = prof.get_total_macs(as_string=False)
    params_saver_ori = prof.get_total_params(as_string=False)
    # prof.print_model_profile()
    prof.end_profile()

    macs_dict = {}
    macs_dict['head'] = []
    macs_dict['saver_model'] = {'name': saver_model_name, 'ori_macs': macs_saver_ori, 'ori_params': params_saver_ori}
    macs_dict['backbone'] = {
        'name': base_model_name,
        'macs': [],
        'macs_ratio': [],
        'params': [],
        'params_ratio': [],
        'ori_macs': macs_ori,
        'ori_params': params_ori
    }
    macs_dict['shared'] = {
        'dim': dmc_params["dim"],
        'macs': [],
        'macs_ratio': [],
    }
    macs_dict['head'].append({'name': f'dmcv2-{saver_model_name}', 'macs': [], 'params': []})

    if 'allowed_exits' in dmc_params and dmc_params['allowed_exits'] is not None:
        exit_positions = dmc_params['allowed_exits'].copy()
        for i in range(len(exit_positions)):
            exit_positions[i] += 1
    else:
        exit_positions = list(range(1, len(interm_feat)))

    exit_positions.append(0)
    print(len(interm_feat), exit_positions)
    for i in tqdm(exit_positions, desc='Profiling'):
        model_main = create_model(interm_feat=True, profiling=i).to(device)
        model_saver = create_saver_model(interm_feat=True).to(device)
        model = dmc_class(model_main, model_saver, **dmc_params).to(device)
        model.eval()
        prof = FlopsProfiler(model)
        prof.start_profile()
        out, interm_feats = model.forward_profiling_backbone(inp)
        macs = prof.get_total_macs(as_string=False)

        # prof.print_model_profile()
        prof.end_profile()
        macs_dict['backbone']['macs'].append(macs)
        macs_dict['backbone']['macs_ratio'].append(macs / macs_ori * 100)

        prof.start_profile()
        interm_feats, saver_model_out = model.forward_profiling_shared(interm_feats)
        macs = prof.get_total_macs(as_string=False)
        prof.end_profile()
        macs_dict['shared']['macs'].append(macs)
        macs_dict['shared']['macs_ratio'].append(macs / macs_ori * 100)

        prof.start_profile()
        single_exit_out = model.forward_profiling_head(interm_feats, saver_model_out)
        macs = prof.get_total_macs(as_string=False)
        prof.end_profile()
        macs_dict['head'][-1]['macs'].append(macs)
        # prof.print_model_profile()

    print(macs_dict['backbone'])
    print('len:', len(macs_dict['backbone']['macs']))

    print(save_path)
    with open(save_path, 'w') as f:
        json.dump(macs_dict, f, indent=4)
    return macs_dict
