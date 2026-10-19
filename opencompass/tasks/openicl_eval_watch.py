import argparse
import copy
import os
import os.path as osp
import random
import sys
import time

from mmengine.config import Config, ConfigDict

from opencompass.registry import TASKS
from opencompass.tasks.openicl_eval import OpenICLEvalTask
from opencompass.utils import (HeartBeatManager, InferStatusManager,
                               dataset_abbr_from_cfg, get_infer_output_path,
                               get_logger, model_abbr_from_cfg)


@TASKS.register_module()
class OpenICLEvalWatchTask(OpenICLEvalTask):
    """Evaluation task that can watch infer progress and run early."""

    name_prefix = 'OpenICLEvalWatch'

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        task_cfg = cfg.get('eval', {}).get('runner', {}).get('task', {})
        self.watch_interval = task_cfg.get('watch_interval', 5.0)
        self.heartbeat = HeartBeatManager(self.work_dir)
        self.heartbeat_timeout = task_cfg.get('heartbeat_timeout', 60.0)
        self.log_interval = task_cfg.get('log_interval', 30.0)

    def get_command(self, cfg_path, template):
        sys.path.append(os.getcwd())
        script_path = __file__
        python = sys.executable
        if self.num_gpus > 1:
            port = random.randint(12000, 32000)
            command = (
                f'{python} -m torch.distributed.run --master_port={port} '
                f'--nproc_per_node {self.num_procs} '
                f'{script_path} {cfg_path}')
        else:
            command = f'{python} {script_path} {cfg_path}'
        return template.format(task_cmd=command)

    def run(self):
        pending = []
        status_index = {}
        # Skip finished tasks
        for model_cfg, dataset_cfgs in zip(self.model_cfgs, self.dataset_cfgs):
            for dataset_cfg in dataset_cfgs:
                out_path = get_infer_output_path(
                    model_cfg, dataset_cfg, osp.join(self.work_dir, 'results'))
                if osp.exists(out_path):
                    continue
                model_abbr = model_abbr_from_cfg(model_cfg)
                dataset_abbr = dataset_abbr_from_cfg(dataset_cfg)
                pending.append((model_cfg, dataset_cfg))
                status_index[(model_abbr, dataset_abbr)] = InferStatusManager(
                    self.work_dir, model_cfg, dataset_cfg)

        if not pending:
            return

        last_log = 0.0
        # Wait for the start of heartbeat.
        time.sleep(10.)
        while pending:
            ready = []
            for model_cfg, dataset_cfg in pending:
                if self._is_ready(model_cfg, dataset_cfg, status_index):
                    ready.append((model_cfg, dataset_cfg))

            if ready:
                for model_cfg, dataset_cfg in ready:
                    dataset_abbr = dataset_abbr_from_cfg(dataset_cfg)
                    self.logger.info(f'Start eval {dataset_abbr}')
                    self.model_cfg = model_cfg
                    self.dataset_cfg = dataset_cfg
                    self.eval_cfg = copy.deepcopy(dataset_cfg.get('eval_cfg'))
                    self.output_column = copy.deepcopy(
                        dataset_cfg['reader_cfg']['output_column'])
                    self._score()
                pending = [item for item in pending if item not in ready]
                continue

            if self.heartbeat.last_heartbeat() > self.heartbeat_timeout:
                for model_cfg, dataset_cfg in pending:
                    model_abbr = model_abbr_from_cfg(model_cfg)
                    dataset_abbr = dataset_abbr_from_cfg(dataset_cfg)
                    self.logger.warning(
                        f'Skip eval for {model_abbr}/{dataset_abbr} '
                        'due to incomplete infer tasks.', )
                return

            now = time.time()
            if now - last_log >= self.log_interval:
                self.logger.info('Waiting infer, pending=%s', len(pending))
                last_log = now

            time.sleep(self.watch_interval)

    def _is_ready(self, model_cfg, dataset_cfg, status_index: dict) -> bool:
        model_abbr = model_abbr_from_cfg(model_cfg)
        dataset_abbr = dataset_abbr_from_cfg(dataset_cfg)
        status = status_index[(model_abbr, dataset_abbr)].get_task_status()
        if status and all(item['status'] == 'done'
                          for item in status.values()):
            return True
        return False


def parse_args():
    parser = argparse.ArgumentParser(description='Score Calculator (watch)')
    parser.add_argument('config', help='Config file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    start_time = time.time()
    inferencer = OpenICLEvalWatchTask(cfg)
    inferencer.run()
    end_time = time.time()
    get_logger().info(f'time elapsed: {end_time - start_time: .2f}s')
