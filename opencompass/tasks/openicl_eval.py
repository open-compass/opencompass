import argparse
import copy
import fnmatch
import math
import os
import os.path as osp
import re
import statistics
import sys
import time
from collections import Counter
from inspect import signature
from typing import List, Optional

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from opencompass.registry import (ICL_EVALUATORS, MODELS, TASKS,
                                  TEXT_POSTPROCESSORS)
from opencompass.tasks.base import BaseTask
from opencompass.utils import (build_dataset_from_cfg, dataset_abbr_from_cfg,
                               get_infer_output_path, get_logger,
                               task_abbr_from_cfg)


def extract_role_pred(s: str, begin_str: Optional[str],
                      end_str: Optional[str]) -> str:
    """Extract the role prediction from the full prediction string. The role
    prediction may be the substring between the begin and end string.

    Args:
        s (str): Full prediction string.
        begin_str (str): The beginning string of the role
        end_str (str): The ending string of the role.

    Returns:
        str: The extracted role prediction.
    """
    start = 0
    end = len(s)

    if begin_str and re.match(r'\s*', begin_str) is None:
        begin_idx = s.find(begin_str)
        if begin_idx != -1:
            start = begin_idx + len(begin_str)

    if end_str and re.match(r'\s*', end_str) is None:
        # TODO: Support calling tokenizer for the accurate eos token
        # and avoid such hardcode
        end_idx = s.find(end_str, start)
        if end_idx != -1:
            end = end_idx

    return s[start:end]


@TASKS.register_module()
class OpenICLEvalTask(BaseTask):
    """OpenICL Evaluation Task.

    This task is used to evaluate the metric between predictions and
    references.
    """

    name_prefix = 'OpenICLEval'
    log_subdir = 'logs/eval'
    output_subdir = 'results'

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        self.logger = get_logger()
        self.num_gpus = max(
            c.get('eval_cfg', {}).get('num_gpus', 0)
            for c in sum(self.dataset_cfgs, []))
        self.dump_details = cfg.get('eval', {}).get('runner', {}).get(
            'task', {}).get('dump_details', False)
        self.cal_extract_rate = cfg.get('eval', {}).get('runner', {}).get(
            'task', {}).get('cal_extract_rate', False)

    def get_command(self, cfg_path, template):
        sys.path.append(os.getcwd())
        script_path = __file__
        python = sys.executable
        command = f'{python} {script_path} {cfg_path}'
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
            pred_dicts = copy.deepcopy(preds)
            preds = {k: [pred.get(k) for pred in preds] for k in preds[0]}

            pred_strs = preds.pop('prediction', None)
            pred_list_flag = pred_strs is not None and isinstance(
                pred_strs[0], list)
            if ('pred_role' in self.eval_cfg
                    and 'meta_template' in self.model_cfg
                    and not MODELS.get(self.model_cfg['type']).is_api):
                # Create a prompt template for role config parsing
                from opencompass.models.base import LMTemplateParser
                parser = LMTemplateParser(self.model_cfg['meta_template'])
                role = parser.roles[self.eval_cfg['pred_role']]
                if sc_size is not None:
                    assert pred_list_flag, (
                        'The prediction for Self-Consistency'
                        'must be list.')
                if pred_list_flag:
                    pred_strs = [[
                        extract_role_pred(_pred, role.get('begin', None),
                                          role.get('end', None))
                        for _pred in pred
                    ] for pred in pred_strs]
                else:
                    pred_strs = [
                        extract_role_pred(pred, role.get('begin', None),
                                          role.get('end', None))
                        for pred in pred_strs
                    ]

            # Postprocess predictions if necessary
            if 'pred_postprocessor' in self.eval_cfg:
                kwargs = self.eval_cfg['pred_postprocessor']
                proc = kwargs.pop('type')
                if isinstance(proc, str):
                    proc = TEXT_POSTPROCESSORS.get(proc)
                if pred_list_flag:
                    pred_strs = [[proc(s, **kwargs) for s in preds]
                                 for preds in pred_strs]
                else:
                    pred_strs = [proc(s, **kwargs) for s in pred_strs]

            # Get majority voting predictions if use self-consistency
            if sc_size is not None:
                pred_strs = [
                    Counter(s).most_common(1)[0][0] for s in pred_strs
                ]

            icl_evaluator = ICL_EVALUATORS.build(self.eval_cfg['evaluator'])
            # need results dir to save other files
            out_path = get_infer_output_path(
                self.model_cfg, self.dataset_cfg,
                osp.join(self.work_dir, 'results'))
            icl_evaluator._out_dir = osp.splitext(out_path)[
                0]  # strip extension

            preds['predictions'] = pred_strs
            preds['references'] = (test_set[self.output_column]
                                   if self.output_column else None)
            preds['test_set'] = test_set
            if 'origin_prompt' not in preds:
                try:
                    preds['origin_prompt'] = [
                        None for _ in range(len(pred_strs))
                    ]
                except TypeError:
                    preds['origin_prompt'] = None
            preds = {
                k: preds[k]
                for k in signature(icl_evaluator.score).parameters
            }
            result = icl_evaluator.score(**preds)

            if self.dump_details:
                details = result.get('details', None)
                try:
                    result['details'] = self.format_details(
                        pred_strs, test_set[self.output_column], details,
                        pred_dicts)
                    result['type'] = result['details'].pop('type', None)
                    if self.cal_extract_rate:
                        # Calculate the extraction success rate for prediction
                        result['extract_rate'] = self.extract_rate(result)

                    if 'PPL' in str(
                            self.dataset_cfg.infer_cfg.inferencer.type):
                        result['correct_bpb'], result['incorrect_bpb'] = \
                            self.calculate_bpb(pred_dicts)
                except Exception as e:
                    self.logger.warning(f'Skip dumping details due to: {e}.')
            else:
                result.pop('details', None)

        if 'error' in result:
            self.logger.error(
                f'Task {task_abbr_from_cfg(self.cfg)}: {result["error"]}')
            return
        else:
            result_wo_details = {
                i: result[i]
                for i in result if i != 'details'
            }
            self.logger.info(
                f'Task {task_abbr_from_cfg(self.cfg)}: {result_wo_details}')

        # Save result
        out_path = get_infer_output_path(self.model_cfg, self.dataset_cfg,
                                         osp.join(self.work_dir, 'results'))
        mkdir_or_exist(osp.split(out_path)[0])
        mmengine.dump(result, out_path, ensure_ascii=False, indent=4)

    def extract_rate(self, results):
        """This function is designed for calculating the extraction rate.

        Args:
            results (dict): The result dict, include the information
        """
        details = results['details']
        details_list = list(details.values())
        invalid_extractions = []
        for item in details_list:
            try:
                invalid_extractions.extend(
                    [item] if not item['predictions'] else [])
            except KeyError as e:
                self.logger.warning(f'Skip {e} due to: {item}')
                raise KeyError
        success_rate = 100 - len(invalid_extractions) / len(details) * 100
        return success_rate

    def format_details(self, predictions, references, details, pred_dicts):
        """This function is responsible for formatting prediction details.

        Args:
            predictions (list): The prediction list.
            references (list): The reference list.
            details (list): Contains the 'pred' 'answer' and 'correct' for each
                sample. Such as `[{'pred': '光荣和ωforce',
                'answers': ['光荣和ω-force', '光荣和ωforce'], 'correct': True}]`
            pred_dicts (list): Contains a list of samples with the original
                prompts. Such as
                `[{'origin_prompt': '根据文章回答问题。你的答案应该尽可能3》…………',
                'prediction': ' 光荣和ω-force\n', 'gold': ['光荣和ω-force']}]`

        Returns:
            list: The formatted prediction details.
        """
        results = {}
        for i in range(len(predictions)):
            ppl_flag = False
            result = {}
            origin_prediction = copy.deepcopy(pred_dicts[i])
            origin_prediction.pop('in-context examples', None)
            origin_prediction.pop('prediction', None)
            keys = copy.deepcopy(list(origin_prediction.keys()))
            for key in keys:
                if key.startswith('label:'):
                    ppl_flag = True
                    origin_prediction[key].pop('testing input', None)
                    new_key = key.replace('label: ', '')
                    origin_prediction[new_key] = origin_prediction.pop(key)
            if ppl_flag:
                results['type'] = 'PPL'
                result['origin_prediction'] = origin_prediction
                result['predictions'] = str(predictions[i])
                result['references'] = str(references[i])
                result['correct'] = str(predictions[i]) == str(references[i])
            elif details is not None:
                results['type'] = 'GEN'
                result['prompt'] = origin_prediction['origin_prompt']
                result['origin_prediction'] = pred_dicts[i]['prediction']
                result['predictions'] = details[i]['pred']
                result['references'] = details[i]['answer']
                result['correct'] = details[i]['correct']
            else:
                results['type'] = 'GEN'
                result['prompt'] = origin_prediction['origin_prompt']
                result['origin_prediction'] = pred_dicts[i]['prediction']
                result['predictions'] = str(predictions[i])
                result['references'] = str(references[i])
            results[str(i)] = result
        return results

    def calculate_bpb(self, pred_dicts: List):
        """This function is used to calculate the BPB (Bits Per Byte) for the
        data. The correct BPB is obtained directly from the values in the
        'predictions' file. The incorrect BPB is the average of the remaining
        BPB values for each sample under different labels after subtracting the
        correct BPB. The calculation of BPB (Bits Per Byte) is similar to PPL,
        with the difference that it computes the additional bits needed on
        average, in terms of character length, to encode the true sequence
        based on the predictions. This calculation involves applying a
        weighting factor based on the ratio of words to characters.

        Args:
            pred_dicts (list): Contains a list of samples with each options
                and BPB scores.

        Returns:
            dict: Contains correct and incorrect bpb.
        """
        incorrect_bpb_list = []
        bpb_list = []
        for pred_dict in pred_dicts:
            preds = {
                key: value
                for key, value in pred_dict.items()
                if key.startswith('label: ')
            }
            values = []
            for item in preds.items():
                values.append(item[1])
            bpbs = [value['BPB'] for value in values]
            incorrect_bpb_list.append(
                (sum(bpbs) - min(bpbs)) / (len(bpbs) - 1))
            bpb_list.append(min(bpbs))

        def filters(origins):
            targets = [target for target in origins if not math.isnan(target)]
            return targets

        mean_incorrect = statistics.mean(filters(incorrect_bpb_list))
        mean_correct = statistics.mean(filters(bpb_list))
        return 100 * mean_correct, 100 * mean_incorrect


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
