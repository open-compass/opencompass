import argparse
import copy
import fnmatch
import os.path as osp
import random
import time
from collections import Counter
from inspect import signature
from typing import Optional

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from opencompass.openicl.icl_evaluator.lm_evaluator import LMEvaluator
from opencompass.registry import (ICL_EVALUATORS, MODELS, TASKS,
                                  TEXT_POSTPROCESSORS)
from opencompass.tasks.base import BaseTask
from opencompass.utils import (build_dataset_from_cfg, dataset_abbr_from_cfg,
                               get_infer_output_path, get_logger,
                               task_abbr_from_cfg)
from opencompass.utils.types import get_type_from_cfg


@TASKS.register_module(force=(__name__ == '__main__'))  # A hack for script run
class OpenICLEvalTask(BaseTask):
    """OpenICL Evaluation Task.

    This task is used to evaluate the metric between predictions and
    references.

    Args:
        cfg (ConfigDict): The configuration of the entire evaluation task.
    """

    name_prefix = 'OpenICLEval'
    log_subdir = 'logs/eval'
    output_subdir = 'results'

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        self.logger = get_logger()
        judge_cfg = cfg.eval.runner.task.get('judge_cfg', {})
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
        for model_cfg, dataset_cfgs in zip(self.model_cfgs, self.dataset_cfgs):
            for dataset_cfg in dataset_cfgs:
                self.model_cfg = model_cfg
                self.dataset_cfg = dataset_cfg

                # Load Dataset
                self.eval_cfg = self.dataset_cfg.get('eval_cfg')
                self.output_column = dataset_cfg['reader_cfg']['output_column']

                # overwrite postprocessor if the model has specified one
                ds_abbr = dataset_abbr_from_cfg(self.dataset_cfg)
                model_postprocessors = self.model_cfg.get(
                    'pred_postprocessor', {})
                for pattern in model_postprocessors.keys():
                    if fnmatch.fnmatch(ds_abbr, pattern):
                        self.eval_cfg[
                            'pred_postprocessor'] = model_postprocessors[
                                pattern]  # noqa
                        break

                out_path = get_infer_output_path(
                    self.model_cfg, self.dataset_cfg,
                    osp.join(self.work_dir, 'results'))
                if osp.exists(out_path):
                    continue
                self._score()

    def _score(self):
        test_set = build_dataset_from_cfg(self.dataset_cfg).test
        # Postprocess dataset if necessary
        if 'dataset_postprocessor' in self.eval_cfg:
            proc = self.eval_cfg['dataset_postprocessor']['type']
            if isinstance(proc, str):
                proc = TEXT_POSTPROCESSORS.get(proc)

            def postprocess(sample):
                s = sample[self.output_column]
                sample[self.output_column] = proc(s)
                return sample

            test_set = test_set.map(postprocess)

        # Load predictions
        filename = get_infer_output_path(
            self.model_cfg, self.dataset_cfg,
            osp.join(self.work_dir, 'predictions'))
        # in case the prediction is partial
        root, ext = osp.splitext(filename)
        partial_filename = root + '_0' + ext

        # Get sc_size if use Self-Consistency
        sc_size = self.eval_cfg.get('sc_size')

        # Get out_path
        out_path = get_infer_output_path(self.model_cfg, self.dataset_cfg,
                                         osp.join(self.work_dir, 'results'))

        if not osp.exists(osp.realpath(filename)) and not osp.exists(
                osp.realpath(partial_filename)):
            result = {'error': 'No predictions found.'}
        else:
            if osp.exists(osp.realpath(filename)):
                preds = mmengine.load(filename)
                preds = [preds[str(i)] for i in range(len(preds))]
            else:
                filename = partial_filename
                preds = []
                i = 1
                while osp.exists(osp.realpath(filename)):
                    sub_preds = mmengine.load(filename)
                    preds.extend(
                        [sub_preds[str(i)] for i in range(len(sub_preds))])
                    filename = root + f'_{i}' + ext
                    i += 1

            preds = {k: [pred.get(k) for pred in preds] for k in preds[0]}

            pred_strs = preds.pop('prediction')

            if ('pred_role' in self.eval_cfg
                    and 'meta_template' in self.model_cfg
                    and not MODELS.get(self.model_cfg['type']).is_api):
                # Create a prompt template for role config parsing
                from opencompass.models.base import LMTemplateParser
                parser = LMTemplateParser(self.model_cfg['meta_template'])
                role = parser.roles[self.eval_cfg['pred_role']]
                if sc_size is not None:
                    for pred in pred_strs:
                        if not isinstance(pred, list):
                            raise TypeError(
                                'The prediction for Self-Consistency'
                                'must be list.')
                        pred_strs.append([
                            self._extract_role_pred(sc_pred,
                                                    role.get('begin', None),
                                                    role.get('end', None))
                            for sc_pred in pred
                        ])
                else:
                    pred_strs = [
                        self._extract_role_pred(pred, role.get('begin', None),
                                                role.get('end', None))
                        for pred in pred_strs
                    ]

            # Postprocess predictions if necessary
            if 'pred_postprocessor' in self.eval_cfg:
                kwargs = self.eval_cfg['pred_postprocessor']
                proc = kwargs.pop('type')
                if isinstance(proc, str):
                    proc = TEXT_POSTPROCESSORS.get(proc)
                if sc_size is not None:
                    pred_strs = [[proc(s, **kwargs) for s in preds]
                                 for preds in pred_strs]
                else:
                    pred_strs = [proc(s, **kwargs) for s in pred_strs]

            # Get majority voting predictions if use self-consistency
            if sc_size is not None:
                pred_strs = [
                    Counter(s).most_common(1)[0][0] for s in pred_strs
                ]

            if get_type_from_cfg(self.eval_cfg['evaluator']) == LMEvaluator:
                if not self.judge_cfg:
                    raise ValueError('Using LMEvaluator in dataset, but '
                                     'missing "eval.runner.task.judge_cfg" '
                                     'as the judge configuration.')
                self.eval_cfg['evaluator']['judge_cfg'] = self.judge_cfg
                self.eval_cfg['evaluator']['dataset_cfg'] = self.dataset_cfg
                self.eval_cfg['evaluator']['output_path'] = out_path
            icl_evaluator = ICL_EVALUATORS.build(self.eval_cfg['evaluator'])
            preds['predictions'] = pred_strs
            preds['references'] = (test_set[self.output_column]
                                   if self.output_column else None)
            preds = {
                k: preds[k]
                for k in signature(icl_evaluator.score).parameters
            }
            result = icl_evaluator.score(**preds)

        if 'error' in result:
            self.logger.error(
                f'Task {task_abbr_from_cfg(self.cfg)}: {result["error"]}')
            return
        else:
            self.logger.info(f'Task {task_abbr_from_cfg(self.cfg)}: {result}')

        # Save result
        mkdir_or_exist(osp.split(out_path)[0])
        mmengine.dump(result,
                      open(out_path, 'w', encoding='utf-8'),
                      file_format='json',
                      ensure_ascii=False,
                      indent=4)

    def _extract_role_pred(self, s: str, begin_str: Optional[str],
                           end_str: Optional[str]) -> str:
        """Extract the role prediction from the full prediction string. The
        role prediction may be the substring between the begin and end string.

        Args:
            s (str): Full prediction string.
            begin_str (str): The beginning string of the role
            end_str (str): The ending string of the role.

        Returns:
            str: The extracted role prediction.
        """
        start = 0
        end = len(s)

        if begin_str:
            begin_idx = s.find(begin_str)
            if begin_idx != -1:
                start = begin_idx + len(begin_str)

        if end_str:
            # TODO: Support calling tokenizer for the accurate eos token
            # and avoid such hardcode
            end_idx = s.find(end_str[:1], start)
            if end_idx != -1:
                end = end_idx

        return s[start:end]


def parse_args():
    parser = argparse.ArgumentParser(description='Score Calculator')
    parser.add_argument('config', help='Config file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    start_time = time.time()
    inferencer = OpenICLEvalTask(cfg)
    inferencer.run()
    end_time = time.time()
    get_logger().info(f'time elapsed: {end_time - start_time:.2f}s')
