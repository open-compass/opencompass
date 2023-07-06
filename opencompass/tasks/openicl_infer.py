import argparse
import os.path as osp
import random
import time
from typing import Any

from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from opencompass.registry import (ICL_INFERENCERS, ICL_PROMPT_TEMPLATES,
                                  ICL_RETRIEVERS, TASKS)
from opencompass.tasks.base import BaseTask
from opencompass.utils import (build_dataset_from_cfg, build_model_from_cfg,
                               get_infer_output_path, get_logger)


@TASKS.register_module(force=(__name__ == '__main__'))  # A hack for script run
class OpenICLInferTask(BaseTask):
    """OpenICL Inference Task.

    This task is used to run the inference process.
    """

    name_prefix = 'OpenICLInfer'
    log_subdir = 'logs/infer'
    output_subdir = 'predictions'

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        run_cfg = self.model_cfgs[0].get('run_cfg', {})
        self.num_gpus = run_cfg.get('num_gpus', 0)
        self.num_procs = run_cfg.get('num_procs', 1)

    def get_command_template(self):
        if self.num_gpus > 0:
            return (f'torchrun --master_port={random.randint(12000, 32000)} '
                    f'--nproc_per_node {self.num_procs} '
                    '{SCRIPT_PATH} {CFG_PATH}')
        else:
            return ('python {SCRIPT_PATH} {CFG_PATH}')

    def run(self):
        for model_cfg, dataset_cfgs in zip(self.model_cfgs, self.dataset_cfgs):
            self.max_out_len = model_cfg.get('max_out_len', None)
            self.batch_size = model_cfg.get('batch_size', None)
            self.model = build_model_from_cfg(model_cfg)

            for dataset_cfg in dataset_cfgs:
                self.model_cfg = model_cfg
                self.dataset_cfg = dataset_cfg
                self.infer_cfg = self.dataset_cfg['infer_cfg']
                self.dataset = build_dataset_from_cfg(self.dataset_cfg)
                out_path = get_infer_output_path(
                    self.model_cfg, self.dataset_cfg,
                    osp.join(self.work_dir, 'predictions'))
                if osp.exists(out_path):
                    continue
                self._inference()

    def _inference(self):

        assert hasattr(self.infer_cfg, 'ice_template') or hasattr(self.infer_cfg, 'prompt_template'), \
            'Both ice_template and prompt_template cannot be None simultaneously.'  # noqa: E501
        if hasattr(self.infer_cfg, 'ice_template'):
            ice_template = ICL_PROMPT_TEMPLATES.build(
                self.infer_cfg['ice_template'])

        if hasattr(self.infer_cfg, 'prompt_template'):
            prompt_template = ICL_PROMPT_TEMPLATES.build(
                self.infer_cfg['prompt_template'])

        retriever_cfg = self.infer_cfg['retriever'].copy()
        retriever_cfg['dataset'] = self.dataset
        retriever = ICL_RETRIEVERS.build(retriever_cfg)

        # set inferencer's default value according to model's config'
        inferencer_cfg = self.infer_cfg['inferencer']
        inferencer_cfg['model'] = self.model
        self._set_default_value(inferencer_cfg, 'max_out_len',
                                self.max_out_len)
        self._set_default_value(inferencer_cfg, 'batch_size', self.batch_size)
        inferencer_cfg['max_seq_len'] = self.model_cfg['max_seq_len']
        inferencer = ICL_INFERENCERS.build(inferencer_cfg)

        out_path = get_infer_output_path(
            self.model_cfg, self.dataset_cfg,
            osp.join(self.work_dir, 'predictions'))
        out_dir, out_file = osp.split(out_path)
        mkdir_or_exist(out_dir)

        if hasattr(self.infer_cfg, 'prompt_template') and \
                hasattr(self.infer_cfg, 'ice_template'):
            inferencer.inference(retriever,
                                 ice_template=ice_template,
                                 prompt_template=prompt_template,
                                 output_json_filepath=out_dir,
                                 output_json_filename=out_file)
        elif hasattr(self.infer_cfg, 'prompt_template'):
            inferencer.inference(retriever,
                                 prompt_template=prompt_template,
                                 output_json_filepath=out_dir,
                                 output_json_filename=out_file)
        else:
            inferencer.inference(retriever,
                                 ice_template=ice_template,
                                 output_json_filepath=out_dir,
                                 output_json_filename=out_file)

    def _set_default_value(self, cfg: ConfigDict, key: str, value: Any):
        if key not in cfg:
            assert value, (f'{key} must be specified!')
            cfg[key] = value


def parse_args():
    parser = argparse.ArgumentParser(description='Model Inferencer')
    parser.add_argument('config', help='Config file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    start_time = time.time()
    inferencer = OpenICLInferTask(cfg)
    inferencer.run()
    end_time = time.time()
    get_logger().info(f'time elapsed: {end_time - start_time:.2f}s')
