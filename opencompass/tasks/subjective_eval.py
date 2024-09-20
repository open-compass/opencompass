# flake8: noqa: E501
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
from opencompass.tasks.openicl_eval import extract_role_pred
from opencompass.utils import (build_dataset_from_cfg, dataset_abbr_from_cfg,
                               deal_with_judge_model_abbr, get_data_path,
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
        judge_cfg = cfg.get('judge_model', None)
        meta_judge_cfg = cfg.get('meta_judge_model', None)
        judge_models = cfg.get('judge_models', None)

        if judge_cfg is None and meta_judge_cfg is None:
            assert judge_cfg is not None, 'Both judge_cfg and meta_judge_cfg are None, but judge_models must be provided.'

        if meta_judge_cfg is not None:
            assert judge_models is not None, 'meta_judge_cfg is provided, but judge_models are missing.'
            judge_cfg = meta_judge_cfg  # Relpace judge_cfg to meta_judge_cfg when it is not None
            self.meta = True
        else:
            self.meta = False
        run_cfg = judge_cfg.get('run_cfg', {})
        self.num_gpus = run_cfg.get('num_gpus', 0)
        self.num_procs = run_cfg.get('num_procs', 1)
        self.judge_cfg = copy.deepcopy(judge_cfg)
        self.judge_models = judge_models
        self.infer_order = cfg.get('infer_order')
        self.given_pred = cfg['datasets'][0][0].get('given_pred', [])

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
                out_path = get_infer_output_path(
                    deal_with_judge_model_abbr(model_cfg, self.judge_cfg,
                                               self.meta), dataset_cfg,
                    osp.join(self.work_dir, 'results'))
                if osp.exists(out_path):
                    continue

                self._score(model_cfg, dataset_cfg, eval_cfg, output_column,
                            self.meta)

    @property
    def name(self) -> str:
        task_name = task_abbr_from_cfg({
            'models': self.model_cfgs,
            'datasets': self.dataset_cfgs
        })
        return self.name_prefix + task_name + \
            '--judge-by--' + model_abbr_from_cfg(self.judge_cfg)

    def _load_model_pred(
        self,
        model_cfg: Union[ConfigDict, List[ConfigDict]],
        dataset_cfg: ConfigDict,
        eval_cfg: ConfigDict,
        given_preds: List[dict],
    ) -> Union[None, List[str]]:
        if isinstance(model_cfg, (tuple, list)):
            return [
                self._load_model_pred(m, dataset_cfg, eval_cfg, given_preds)
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
            last_underscore_index = root.rfind('_')
            root = root[:last_underscore_index]
            filename = root + ext
        # If take SubjectNaivePartition, get filename
        else:
            filename = get_infer_output_path(
                model_cfg, dataset_cfg, osp.join(self.work_dir, 'predictions'))
        for given_pred in given_preds:
            abbr = given_pred['abbr']
            path = given_pred['path']
            path = get_data_path(path, local_mode=True)
            if abbr == model_cfg['abbr']:
                filename = osp.join(path, osp.basename(filename))
        # Get partition name
        root, ext = osp.splitext(filename)
        partial_filename = root + '_0' + ext
        # If no predictions get in predictions dir
        assert osp.exists(filename) or osp.exists(
            osp.realpath(partial_filename)
        ), 'No predictions found for {filename} and {partial_filename}'.format(
            filename=filename, partial_filename=partial_filename)

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
                and not MODELS.get(model_cfg['type']).is_api
                and isinstance(pred_strs[0], str)):
            # Create a prompt template for role config parsing
            from opencompass.models.base import LMTemplateParser
            parser = LMTemplateParser(model_cfg['meta_template'])
            role = parser.roles[eval_cfg['pred_role']]
            pred_strs = [
                extract_role_pred(pred, role.get('begin', None),
                                  role.get('end', None)) for pred in pred_strs
            ]

        # Postprocess predictions if necessary
        ds_abbr = dataset_abbr_from_cfg(dataset_cfg)
        model_postprocessors = model_cfg.get('pred_postprocessor', {})
        pred_postprocessor = None
        for pattern in model_postprocessors.keys():
            if fnmatch.fnmatch(ds_abbr, pattern):
                pred_postprocessor = model_postprocessors[pattern]
                break
        if 'pred_postprocessor' in eval_cfg['evaluator'] or pred_postprocessor:
            kwargs = pred_postprocessor or eval_cfg['evaluator'][
                'pred_postprocessor']
            proc = TEXT_POSTPROCESSORS.get(kwargs.pop('type'))
            self.logger.info('Get postprocessor {postprocessor}.')
            pred_strs = [proc(s, **kwargs) for s in pred_strs]
        else:
            self.logger.info('No postprocessor found.')

        return {
            'model_name': model_abbr_from_cfg(model_cfg),
            'model_preds': pred_strs
        }

    def _load_model_judgements(
        self,
        model_cfg: Union[ConfigDict, List[ConfigDict]],
        dataset_cfg: ConfigDict,
        eval_cfg: ConfigDict,
        judge_cfg: Union[ConfigDict, List[ConfigDict]],
    ) -> Union[None, List[str]]:

        if isinstance(judge_cfg, (tuple, list)):
            return [
                self._load_model_judgements(model_cfg, dataset_cfg, eval_cfg,
                                            j) for j in judge_cfg
            ]

        pred_strs = None
        model_cfg = [model_cfg] if isinstance(model_cfg,
                                              ConfigDict) else model_cfg
        # There will be 5 situations, so we need to deal with them
        # 1.There are no partitions in infer and judge stage
        # 2.No partition in infer stage, but use partition in judge stage
        # 3.Use partition in infer stage, but not use partition in judge stage
        # 4.Use both partition, with same partition size
        # 5.Use both partition, but different partition size

        # If take SubjectSizePartition, get new filename without _0
        if 'test_range' in dataset_cfg['reader_cfg']:
            filename = get_infer_output_path(
                deal_with_judge_model_abbr([m for m in model_cfg], judge_cfg),
                dataset_cfg, osp.join(self.work_dir, 'results'))
            root, ext = osp.splitext(filename)
            last_underscore_index = root.rfind('_')
            root = root[:last_underscore_index]
            filename = root + ext
        # If take SubjectNaivePartition, get filename
        else:
            filename = get_infer_output_path(
                deal_with_judge_model_abbr([m for m in model_cfg], judge_cfg),
                dataset_cfg, osp.join(self.work_dir, 'results'))
        # Get partition name
        root, ext = osp.splitext(filename)
        partial_filename = root + '_0' + ext

        # If no predictions get in predictions dir
        if not osp.exists(osp.realpath(filename)) and not osp.exists(
                osp.realpath(partial_filename)):
            return {'error': 'No judgements found.'}
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
        # Get all judgements in pred_strs
        # If take SubjectSizePartition, get new pred_strs based on test_range
        if 'test_range' in dataset_cfg['reader_cfg']:
            test_range = dataset_cfg['reader_cfg']['test_range']
            if self.infer_order == 'double':
                # When set infer_order as double, we need to select the judgements to meet the predctions which will be doubled later
                start = 0
                end = None
                pred_strs_length = len(pred_strs)
                # Split the string by the ':', the test_range is a string shapes like '[0:15]'
                parts = test_range.strip('[]').split(':')
                # Check if the start index is provided
                if parts[0]:
                    start = int(parts[0])
                # Check if the end index is provided
                if len(parts) > 1 and parts[1]:
                    end = int(parts[1])
                else:
                    # If the end is not provided, determine the default end based on the length of 'pred_strs'
                    end = int(pred_strs_length / 2)
                assert pred_strs_length % 2 == 0, "Since you have set the infer_order as 'double', the length of 'pred_strs' must be even."
                assert end <= pred_strs_length / 2, "The 'end' value must not exceed half of the 'pred_strs' length."
                # Reset the newly start and end
                start *= 2
                end *= 2
                pred_strs = eval('pred_strs[' + str(start) + ':' + str(end) +
                                 ']')
            else:
                pred_strs = eval('pred_strs' + test_range)
        # If take SubjectNaivePartition, get all pred_strs
        else:
            pred_strs = pred_strs
        if ('pred_role' in eval_cfg and 'meta_template' in judge_cfg
                and not MODELS.get(judge_cfg['type']).is_api
                and isinstance(pred_strs[0], str)):
            # Create a prompt template for role config parsing
            from opencompass.models.base import LMTemplateParser
            parser = LMTemplateParser(judge_cfg['meta_template'])
            role = parser.roles[eval_cfg['pred_role']]
            pred_strs = [
                extract_role_pred(pred, role.get('begin', None),
                                  role.get('end', None)) for pred in pred_strs
            ]

        # Postprocess predictions if necessary
        ds_abbr = dataset_abbr_from_cfg(dataset_cfg)
        model_postprocessors = judge_cfg.get('pred_postprocessor', {})
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
            'model_name': model_abbr_from_cfg(judge_cfg),
            'model_preds': pred_strs
        }

    def _score(self,
               model_cfg,
               dataset_cfg,
               eval_cfg,
               output_column,
               meta=False):
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
        out_path = get_infer_output_path(
            deal_with_judge_model_abbr(model_cfg, self.judge_cfg, self.meta),
            dataset_cfg, osp.join(self.work_dir, 'results'))
        if meta:
            model_preds = self._load_model_pred(model_cfg, dataset_cfg,
                                                eval_cfg, self.given_pred)
            model_judges = self._load_model_judgements(model_cfg, dataset_cfg,
                                                       eval_cfg,
                                                       self.judge_models)
        else:
            model_preds = self._load_model_pred(model_cfg, dataset_cfg,
                                                eval_cfg, self.given_pred)
            model_judges = None
        if not self.judge_cfg:
            raise ValueError('missing "eval.judge_cfg"')
        eval_cfg['evaluator']['judge_cfg'] = self.judge_cfg
        eval_cfg['evaluator']['dataset_cfg'] = dataset_cfg
        eval_cfg['evaluator']['output_path'] = out_path
        icl_evaluator = ICL_EVALUATORS.build(eval_cfg['evaluator'])
        references = (test_set[output_column] if output_column else None)
        if 'error' not in model_preds:
            result = icl_evaluator.score(predictions=model_preds,
                                         judgements=model_judges,
                                         references=references,
                                         meta=meta,
                                         infer_order=self.infer_order)
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
                if isinstance(model, ConfigDict):
                    model = (model, )
                if self.meta:
                    model += ({
                        'abbr':
                        'summarized-by--' + model_abbr_from_cfg(self.judge_cfg)
                    }, )
                else:
                    model += ({
                        'abbr':
                        'judged-by--' + model_abbr_from_cfg(self.judge_cfg)
                    }, )
                output_paths.append(
                    get_infer_output_path(
                        model, dataset,
                        osp.join(self.work_dir, self.output_subdir),
                        file_extension))
                model = model[:-1]
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
