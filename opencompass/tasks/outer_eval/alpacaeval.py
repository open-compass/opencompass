# flake8: noqa: E501
import copy
import json
import os
import os.path as osp

import mmengine
from mmengine.config import Config, ConfigDict

from opencompass.tasks.base import BaseTask
from opencompass.utils import (build_dataset_from_cfg, get_infer_output_path,
                               get_logger)


class PredictionMerger:
    """"""

    def __init__(self, cfg: ConfigDict) -> None:

        self.cfg = cfg
        self.model_cfg = copy.deepcopy(self.cfg['model'])
        self.dataset_cfg = copy.deepcopy(self.cfg['dataset'])

        self.work_dir = self.cfg.get('work_dir')

    def run(self):
        filename = get_infer_output_path(
            self.model_cfg, self.dataset_cfg,
            osp.join(self.work_dir, 'predictions'))
        root, ext = osp.splitext(filename)
        alpaca_format_filename = root + '_alpaca' + ext
        partial_filename = root + '_0' + ext

        if osp.exists(osp.realpath(alpaca_format_filename)):
            return

        if not osp.exists(osp.realpath(partial_filename)) and not osp.exists(
                osp.realpath(filename)):
            print(f'{filename} not found')
            return

        # Load predictions
        partial_filenames = []
        if osp.exists(osp.realpath(filename)):
            preds = mmengine.load(filename)
        else:
            preds, offset = {}, 0
            i = 1
            while osp.exists(osp.realpath(partial_filename)):
                partial_filenames.append(osp.realpath(partial_filename))
                _preds = mmengine.load(partial_filename)
                partial_filename = root + f'_{i}' + ext
                i += 1
                for _o in range(len(_preds)):
                    preds[str(offset)] = _preds[str(_o)]
                    offset += 1

        dataset = build_dataset_from_cfg(self.dataset_cfg)
        if len(preds) != len(dataset.test):
            print('length mismatch')
            return

        with open(
                osp.realpath(osp.join(self.dataset_cfg['path'],
                                      'example.json')), 'r') as f:
            data_format = json.load(f)

        for idx in range(len(preds)):
            data_format[idx]['output'] = preds[str(idx)]['prediction']
            data_format[idx]['generator'] = self.model_cfg['abbr']

        print(f'Convert to {alpaca_format_filename}')
        with open(alpaca_format_filename, 'w', encoding='utf-8') as f:
            json.dump(data_format, f, indent=4, ensure_ascii=False)


class AlpacaEvalTask(BaseTask):
    """Subjective Evaluation Task.

    This task is used to evaluate the metric between predictions and
    references.

    Args:
        cfg (ConfigDict): The configuration of the entire evaluation task.
    """

    name_prefix = 'SubjectiveEval'
    log_subdir = 'logs/eval'
    output_subdir = 'results'

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        self.logger = get_logger()
        judge_cfg = cfg.eval.runner.task.get('judge_cfg', {})
        assert type(judge_cfg) == ConfigDict
        run_cfg = judge_cfg.get('run_cfg', {})
        self.num_gpus = run_cfg.get('num_gpus', 0)
        self.num_procs = run_cfg.get('num_procs', 1)
        self.judge_cfg = copy.deepcopy(judge_cfg)

    def get_command(self, cfg_path, template):
        """Get the command template for the task.

        Args:
            cfg_path (str): The path to the config file of the task.
            template (str): The template which have '{task_cmd}' to format
                the command.
        """
        # script_path = __file__
        alpaca_cfg = self.judge_cfg.get('config', None)
        api_key = self.judge_cfg.get('key', None)
        base_url = self.judge_cfg.get('base_url', None)
        assert alpaca_cfg is not None
        all_cfg = Config.fromfile(cfg_path)
        model_cfg = all_cfg['models']
        dataset_cfg = all_cfg['datasets'][0][0]
        work_dir = osp.realpath(all_cfg['work_dir'])
        for m_cfg in model_cfg:
            PredictionMerger({
                'model': m_cfg,
                'dataset': dataset_cfg,
                'work_dir': work_dir
            }).run()
            filename = get_infer_output_path(m_cfg, dataset_cfg,
                                             osp.join(work_dir, 'predictions'))
            root, ext = osp.splitext(filename)
            alpaca_format_filename = root + '_alpaca' + ext
            output_path = osp.join(work_dir, 'results', m_cfg['abbr'])
            if not osp.exists(output_path):
                os.makedirs(output_path)
            caching_path = osp.join(output_path, 'tmp_annotations.json')
            command = ''
            if api_key is not None:
                command += f'export OPENAI_API_KEY={api_key}; '
            else:
                api_key = os.environ.get('OPENAI_API_KEY', '').split(',')[0]
                if api_key:
                    command += f'export OPENAI_API_KEY={api_key}; '
            if base_url is not None:
                command += f'export OPENAI_BASE_URL={base_url}; '
            command += f'alpaca_eval --model_outputs {alpaca_format_filename} --annotators_config {alpaca_cfg} --output_path {output_path} --caching_path {caching_path};'
            return template.format(task_cmd=command)

    def run(self):
        # model_cfg can be a list of model configs
        pass
