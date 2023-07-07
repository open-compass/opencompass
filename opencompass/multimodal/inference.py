import argparse
import json
import os
import os.path as osp
import time
from typing import Sequence

import mmengine
import torch
import torch.distributed as dist
from mmengine.config import Config, DictAction
from mmengine.device import get_device
from mmengine.dist import init_dist
from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.model.wrappers import MMDistributedDataParallel
from mmengine.runner import Runner
from mmengine.utils import track_iter_progress

from opencompass.registry import MODELS


def parse_args():

    parser = argparse.ArgumentParser(description='Benchmark a model')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'])
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--work-dir', help='the dir to save logs and models')

    return parser.parse_args()


def build_model(cfg):
    model = MODELS.build(cfg['model'])
    load_from = cfg.get('load_from', None)
    if load_from is not None:
        state_dict = torch.load(cfg['load_from'], map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        msg = model.load_state_dict(state_dict, strict=False)
        print_log(msg)
    model.to(get_device())
    if dist.is_initialized():
        model = MMDistributedDataParallel(
            model,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
    return model


def main(args):
    # init distributed mode
    if args.launcher != 'none':
        init_dist(args.launcher)

    # we use the config system from MMEngine
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # create time stamps
    timestamp = torch.tensor(time.time(), dtype=torch.float64)
    timestamp = time.strftime('%Y%m%d_%H%M%S',
                              time.localtime(timestamp.item()))
    cfg.work_dir = osp.join(osp.abspath(cfg.work_dir), timestamp)
    mmengine.mkdir_or_exist(cfg.work_dir)
    config_filename = 'config.py'
    cfg.dump(osp.join(cfg.work_dir, config_filename))

    # build dataloader
    dataloader = Runner.build_dataloader(cfg.dataloader)

    # build evaluator
    evaluator = Evaluator(cfg.evaluator)

    # build model
    model = build_model(cfg)

    for batch in track_iter_progress(dataloader):
        if dist.is_initialized():
            data_samples = model.module.generate(batch)
        else:
            data_samples = model.generate(batch)
        if not isinstance(data_samples, Sequence):
            data_samples = [data_samples]
        evaluator.process(data_samples)

    metrics = evaluator.evaluate(len(dataloader.dataset))
    metrics_file = osp.join(cfg.work_dir, 'res.log')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    args = parse_args()
    main(args)
