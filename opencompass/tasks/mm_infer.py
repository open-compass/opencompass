import argparse
import json
import os
import os.path as osp
import random
import time
from typing import List, Sequence

import mmengine
import torch
import torch.distributed as dist
from mmengine.config import Config, ConfigDict
from mmengine.device import get_device
from mmengine.dist import init_dist
from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.model.wrappers import MMDistributedDataParallel
from mmengine.runner import Runner
from mmengine.utils import track_iter_progress

from opencompass.registry import MM_MODELS, TASKS
from opencompass.utils import get_logger


def build_model(cfg):
    model = MM_MODELS.build(cfg['model'])
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


@TASKS.register_module(force=(__name__ == '__main__'))  # A hack for script run
class MultimodalInferTask:
    """Multimodal Inference Task.

    This task is used to run the inference process.
    """

    def __init__(self, cfg: ConfigDict):
        self.num_gpus = cfg.get('num_gpus', 0)
        self.num_procs = cfg.get('num_procs', 1)
        self.dataloader = cfg.get('dataset')
        self.model = cfg.get('model')
        self.evaluator = cfg.get('evaluator')
        self.cfg = cfg
        self.logger = get_logger()

    @property
    def name(self) -> str:
        model_name = self.model['type']
        dataset_name = self.dataloader['dataset']['type']
        evaluator_name = self.evaluator[0]['type']
        return f'{model_name}-{dataset_name}-{evaluator_name}'

    def get_log_path(self, file_extension: str = 'json') -> str:
        """Get the path to the log file.

        Args:
            file_extension (str): The file extension of the log file.
                Default: 'json'.
        """
        model_name = self.model['type']
        dataset_name = self.dataloader['dataset']['type']
        evaluator_name = self.evaluator[0]['type']

        return osp.join(self.cfg.work_dir, model_name, dataset_name,
                        f'{evaluator_name}.{file_extension}')

    def get_output_paths(self, file_extension: str = 'json') -> List[str]:
        """Get the path to the output file.

        Args:
            file_extension (str): The file extension of the log file.
                Default: 'json'.
        """
        model_name = self.model['type']
        dataset_name = self.dataloader['dataset']['type']
        evaluator_name = self.evaluator[0]['type']

        return [
            osp.join(self.cfg.work_dir, model_name, dataset_name,
                     f'{evaluator_name}.{file_extension}')
        ]

    def get_command(self, cfg_path, template):
        """Get the command template for the task.

        Args:
            cfg_path (str): The path to the config file of the task.
            template (str): The template which have '{task_cmd}' to format
                the command.
        """
        script_path = __file__
        if self.num_gpus > 0:
            port = random.randint(12000, 32000)
            command = (f'torchrun --master_port={port} '
                       f'--nproc_per_node {self.num_procs} '
                       f'{script_path} {cfg_path}')
        else:
            command = f'python {script_path} {cfg_path}'

        return template.format(task_cmd=command)

    def run(self):
        # only support slurm, pytorch, mpi
        init_dist(self.cfg.launcher)
        self.logger.info(f'Task {self.name}')
        # build dataloader
        dataloader = Runner.build_dataloader(self.dataloader)
        # build model
        model = build_model(self.cfg)
        model.eval()
        # build evaluator
        evaluator = Evaluator(self.evaluator)

        for batch in track_iter_progress(dataloader):
            if dist.is_initialized():
                data_samples = model.module.forward(batch)
            else:
                data_samples = model.forward(batch)
            if not isinstance(data_samples, Sequence):
                data_samples = [data_samples]
            evaluator.process(data_samples)

        metrics = evaluator.evaluate(len(dataloader.dataset))
        metrics_file = self.get_output_paths()[0]
        mmengine.mkdir_or_exist(osp.split(metrics_file)[0])
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)


def parse_args():
    parser = argparse.ArgumentParser(description='Model Inferencer')
    parser.add_argument('config', help='Config file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    start_time = time.time()
    inferencer = MultimodalInferTask(cfg)
    inferencer.run()
    end_time = time.time()
    get_logger().info(f'time elapsed: {end_time - start_time:.2f}s')
