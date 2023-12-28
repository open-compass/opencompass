import argparse
import copy
import fnmatch
import os.path as osp
import random
import time
from typing import List, Optional, Union

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from opencompass.registry import ICL_EVALUATORS, MODELS, TEXT_POSTPROCESSORS
from opencompass.tasks.base import BaseTask
from opencompass.utils import (build_dataset_from_cfg, dataset_abbr_from_cfg,
                               get_infer_output_path, get_logger,
                               model_abbr_from_cfg, task_abbr_from_cfg)


class SubjectiveEvalTask(BaseTask):
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
        if type(judge_cfg) != ConfigDict:
            print('*' * 100)
            print('Due to different Judge model needs different summarizer and'
                  " prompts, we don't support multi judge model evaluation at "
                  'one time, please do not use list to set your judge cfg, jus'
                  't use a dict or list[0] should be fine. If you want to eval'
                  'uation multi judge model in one script, we suggest you to u'
                  'se a bash or bat script to start multi configs evaluation!')
            print('*' * 100)
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
        # model_cfg can be a list of model configs
        for model_cfg, dataset_cfgs in zip(self.model_cfgs, self.dataset_cfgs):
            for dataset_cfg in dataset_cfgs:
                # Load Dataset
                eval_cfg = dataset_cfg.get('eval_cfg')
                output_column = dataset_cfg['reader_cfg']['output_column']
                if type(model_cfg) == ConfigDict:
                    model_cfg = (model_cfg, )
                model_cfg += ({
                    'abbr':
                    'judged-by--' + model_abbr_from_cfg(self.judge_cfg)
                }, )
                out_path = get_infer_output_path(
                    model_cfg, dataset_cfg, osp.join(self.work_dir, 'results'))
                if osp.exists(out_path):
                    continue
                self._score(model_cfg, dataset_cfg, eval_cfg, output_column)

    def _load_model_pred(self, model_cfg: Union[ConfigDict, List[ConfigDict]],
                         dataset_cfg: ConfigDict,
                         eval_cfg: ConfigDict) -> Union[None, List[str]]:
        if isinstance(model_cfg, (tuple, list)):
            return [
                self._load_model_pred(m, dataset_cfg, eval_cfg)
                for m in model_cfg
            ]

        pred_strs = None

        # There will be 5 situations, so we need to deal with them
        # 1.There are no partitions in infer and judge stage
        # 2.No partition in infer stage, but use partition in judge stage
        # 3.Use partition in infer stage, but not use partition in judge stage
        # 4.Use both partition, with same partition size
        # 5.Use both partition, but different partition size

        # If take SubjectSizePartition, get new filename without _0
        if 'test_range' in dataset_cfg['reader_cfg']:
            filename = get_infer_output_path(
                model_cfg, dataset_cfg, osp.join(self.work_dir, 'predictions'))
            root, ext = osp.splitext(filename)
            filename = root[:-2] + ext
        # If take SubjectNaivePartition, get filename
        else:
            filename = get_infer_output_path(
                model_cfg, dataset_cfg, osp.join(self.work_dir, 'predictions'))

        # Get partition name
        root, ext = osp.splitext(filename)
        partial_filename = root + '_0' + ext

        # If no predictions get in predictions dir
        if not osp.exists(osp.realpath(filename)) and not osp.exists(
                osp.realpath(partial_filename)):
            return {'error': 'No predictions found.'}
        else:
            # If use Naive partition in infer stage
            if osp.exists(osp.realpath(filename)):
                preds = mmengine.load(filename)
                pred_strs = [
                    preds[str(i)]['prediction'] for i in range(len(preds))
                ]
            # If use Size partition in infer stage
            else:
                filename = partial_filename
                pred_strs = []
                i = 1
                while osp.exists(osp.realpath(filename)):
                    preds = mmengine.load(filename)
                    filename = root + f'_{i}' + ext
                    i += 1
                    pred_strs += [
                        preds[str(i)]['prediction'] for i in range(len(preds))
                    ]
        # Get all predictions in pred_strs

        # If take SubjectSizePartition, get new pred_strs based on test_range
        if 'test_range' in dataset_cfg['reader_cfg']:
            test_range = dataset_cfg['reader_cfg']['test_range']
            pred_strs = eval('pred_strs' + test_range)
        # If take SubjectNaivePartition, get all pred_strs
        else:
            pred_strs = pred_strs

        if ('pred_role' in eval_cfg and 'meta_template' in model_cfg
                and not MODELS.get(model_cfg['type']).is_api):
            # Create a prompt template for role config parsing
            from opencompass.models.base import LMTemplateParser
            parser = LMTemplateParser(model_cfg['meta_template'])
            role = parser.roles[eval_cfg['pred_role']]
            pred_strs = [
                self._extract_role_pred(pred, role.get('begin', None),
                                        role.get('end', None))
                for pred in pred_strs
            ]

        # Postprocess predictions if necessary
        ds_abbr = dataset_abbr_from_cfg(dataset_cfg)
        model_postprocessors = model_cfg.get('pred_postprocessor', {})
        pred_postprocessor = None
        for pattern in model_postprocessors.keys():
            if fnmatch.fnmatch(ds_abbr, pattern):
                pred_postprocessor = model_postprocessors[pattern]
                break
        if 'pred_postprocessor' in eval_cfg or pred_postprocessor:
            kwargs = pred_postprocessor or eval_cfg['pred_postprocessor']
            proc = TEXT_POSTPROCESSORS.get(kwargs.pop('type'))
            pred_strs = [proc(s, **kwargs) for s in pred_strs]

        return {
            'model_name': model_abbr_from_cfg(model_cfg),
            'model_preds': pred_strs
        }

    def _score(self, model_cfg, dataset_cfg, eval_cfg, output_column):
        test_set = build_dataset_from_cfg(dataset_cfg).test
        # Postprocess dataset if necessary
        if 'dataset_postprocessor' in eval_cfg:
            proc = TEXT_POSTPROCESSORS.get(
                eval_cfg['dataset_postprocessor']['type'])

            def postprocess(sample):
                s = sample[output_column]
                sample[output_column] = proc(s)
                return sample

            test_set = test_set.map(postprocess)
        # Get out_path
        out_path = get_infer_output_path(model_cfg, dataset_cfg,
                                         osp.join(self.work_dir, 'results'))
        new_model_cfg = []
        for m_cfg in model_cfg:
            if len(m_cfg) > 1:
                new_model_cfg.append(m_cfg)
        if len(new_model_cfg) == 1:
            new_model_cfg = new_model_cfg[0]
        model_preds = self._load_model_pred(new_model_cfg, dataset_cfg,
                                            eval_cfg)
        if not self.judge_cfg:
            raise ValueError('missing "eval.runner.task.judge_cfg"')
        eval_cfg['evaluator']['judge_cfg'] = self.judge_cfg
        eval_cfg['evaluator']['dataset_cfg'] = dataset_cfg
        eval_cfg['evaluator']['output_path'] = out_path
        icl_evaluator = ICL_EVALUATORS.build(eval_cfg['evaluator'])
        references = (test_set[output_column] if output_column else None)

        if 'error' not in model_preds:
            result = icl_evaluator.score(predictions=model_preds,
                                         references=references)
        else:
            result = model_preds

        if 'error' in result:
            self.logger.error(
                f'Task {task_abbr_from_cfg(self.cfg)}: {result["error"]}')
            return
        else:
            self.logger.info(
                f'Task {task_abbr_from_cfg(self.cfg)}')  #: {result}')

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

    def get_output_paths(self, file_extension: str = 'json') -> List[str]:
        """Get the paths to the output files. Every file should exist if the
        task succeeds.

        Args:
            file_extension (str): The file extension of the output files.
                Default: 'json'.
        """
        output_paths = []
        for model, datasets in zip(self.model_cfgs, self.dataset_cfgs):
            for dataset in datasets:
                if type(model) == ConfigDict:
                    model = (model, )
                model += ({
                    'abbr':
                    'judged-by--' + model_abbr_from_cfg(self.judge_cfg)
                }, )
                output_paths.append(
                    get_infer_output_path(
                        model, dataset,
                        osp.join(self.work_dir, self.output_subdir),
                        file_extension))
        return output_paths


def parse_args():
    parser = argparse.ArgumentParser(description='Score Calculator')
    parser.add_argument('config', help='Config file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    start_time = time.time()
    inferencer = SubjectiveEvalTask(cfg)
    inferencer.run()
    end_time = time.time()
    get_logger().info(f'time elapsed: {end_time - start_time:.2f}s')
