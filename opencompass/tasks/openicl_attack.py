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
                               get_infer_output_path, get_logger,
                               task_abbr_from_cfg)


@TASKS.register_module()
class OpenICLAttackTask(BaseTask):
    """OpenICL Inference Task.

    This task is used to run the inference process.
    """

    name_prefix = 'OpenICLAttack'
    log_subdir = 'logs/attack'
    output_subdir = 'attack'

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        run_cfg = self.model_cfgs[0].get('run_cfg', {})
        self.num_gpus = run_cfg.get('num_gpus', 0)
        self.num_procs = run_cfg.get('num_procs', 1)
        self.logger = get_logger()

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

    def prompt_selection(self, inferencer, prompts):
        prompt_dict = {}

        for prompt in prompts:
            acc = inferencer.predict(prompt)
            prompt_dict[prompt] = acc
            self.logger.info('{:.2f}, {}\n'.format(acc * 100, prompt))

        sorted_prompts = sorted(prompt_dict.items(),
                                key=lambda x: x[1],
                                reverse=True)
        return sorted_prompts

    def run(self):
        self.logger.info(f'Task {task_abbr_from_cfg(self.cfg)}')
        for model_cfg, dataset_cfgs in zip(self.model_cfgs, self.dataset_cfgs):
            self.max_out_len = model_cfg.get('max_out_len', None)
            self.batch_size = model_cfg.get('batch_size', None)
            self.model = build_model_from_cfg(model_cfg)

            for dataset_cfg in dataset_cfgs:
                self.model_cfg = model_cfg
                self.dataset_cfg = dataset_cfg
                self.infer_cfg = self.dataset_cfg['infer_cfg']
                self.dataset = build_dataset_from_cfg(self.dataset_cfg)
                self.sub_cfg = {
                    'models': [self.model_cfg],
                    'datasets': [[self.dataset_cfg]],
                }
                out_path = get_infer_output_path(
                    self.model_cfg, self.dataset_cfg,
                    osp.join(self.work_dir, 'attack'))
                if osp.exists(out_path):
                    continue
                self._inference()

    def _inference(self):
        self.logger.info(
            f'Start inferencing {task_abbr_from_cfg(self.sub_cfg)}')

        assert hasattr(self.infer_cfg, 'ice_template') or hasattr(self.infer_cfg, 'prompt_template'), \
            'Both ice_template and prompt_template cannot be None simultaneously.'  # noqa: E501
        ice_template = None
        if hasattr(self.infer_cfg, 'ice_template'):
            ice_template = ICL_PROMPT_TEMPLATES.build(
                self.infer_cfg['ice_template'])

        prompt_template = None
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
        inferencer_cfg['dataset_cfg'] = self.dataset_cfg
        inferencer = ICL_INFERENCERS.build(inferencer_cfg)

        out_path = get_infer_output_path(self.model_cfg, self.dataset_cfg,
                                         osp.join(self.work_dir, 'attack'))
        out_dir, out_file = osp.split(out_path)
        mkdir_or_exist(out_dir)

        from config import LABEL_SET
        from prompt_attack.attack import create_attack
        from prompt_attack.goal_function import PromptGoalFunction

        inferencer.retriever = retriever
        inferencer.prompt_template = prompt_template
        inferencer.ice_template = ice_template
        inferencer.output_json_filepath = out_dir
        inferencer.output_json_filename = out_file
        goal_function = PromptGoalFunction(
            inference=inferencer,
            query_budget=self.cfg['attack'].query_budget,
            logger=self.logger,
            model_wrapper=None,
            verbose='True')
        if self.cfg['attack']['dataset'] not in LABEL_SET:
            # set default
            self.cfg['attack']['dataset'] = 'mmlu'
        attack = create_attack(self.cfg['attack'], goal_function)

        prompts = self.infer_cfg['inferencer']['original_prompt_list']
        sorted_prompts = self.prompt_selection(inferencer, prompts)
        if True:
            # if args.prompt_selection:
            for prompt, acc in sorted_prompts:
                self.logger.info('Prompt: {}, acc: {:.2f}%\n'.format(
                    prompt, acc * 100))
                with open(out_dir + 'attacklog.txt', 'a+') as f:
                    f.write('Prompt: {}, acc: {:.2f}%\n'.format(
                        prompt, acc * 100))

        for init_prompt, init_acc in sorted_prompts[:self.cfg['attack'].
                                                    prompt_topk]:
            if init_acc > 0:
                init_acc, attacked_prompt, attacked_acc, dropped_acc = attack.attack(  # noqa
                    init_prompt)
                self.logger.info('Original prompt: {}'.format(init_prompt))
                self.logger.info('Attacked prompt: {}'.format(
                    attacked_prompt.encode('utf-8')))
                self.logger.info(
                    'Original acc: {:.2f}%, attacked acc: {:.2f}%, dropped acc: {:.2f}%'  # noqa
                    .format(init_acc * 100, attacked_acc * 100,
                            dropped_acc * 100))
                with open(out_dir + 'attacklog.txt', 'a+') as f:
                    f.write('Original prompt: {}\n'.format(init_prompt))
                    f.write('Attacked prompt: {}\n'.format(
                        attacked_prompt.encode('utf-8')))
                    f.write(
                        'Original acc: {:.2f}%, attacked acc: {:.2f}%, dropped acc: {:.2f}%\n\n'  # noqa
                        .format(init_acc * 100, attacked_acc * 100,
                                dropped_acc * 100))
            else:
                with open(out_dir + 'attacklog.txt', 'a+') as f:
                    f.write('Init acc is 0, skip this prompt\n')
                    f.write('Original prompt: {}\n'.format(init_prompt))
                    f.write('Original acc: {:.2f}% \n\n'.format(init_acc *
                                                                100))

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
    inferencer = OpenICLAttackTask(cfg)
    inferencer.run()
    end_time = time.time()
    get_logger().info(f'time elapsed: {end_time - start_time:.2f}s')
