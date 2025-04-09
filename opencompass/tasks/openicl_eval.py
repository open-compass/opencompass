import argparse
import copy
import math
import os
import os.path as osp
import random
import statistics
import sys
import time
from inspect import signature
from typing import List

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from opencompass.registry import (ICL_EVALUATORS, MODELS, TASKS,
                                  TEXT_POSTPROCESSORS)
from opencompass.tasks.base import BaseTask, extract_role_pred
from opencompass.utils import (build_dataset_from_cfg, get_infer_output_path,
                               get_logger)


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
            max(
                c.get('eval_cfg', {}).get('num_gpus', 0),
                c.get('eval_cfg', {}).get('evaluator', {}).get(
                    'judge_cfg', {}).get('run_cfg', {}).get('num_gpus', 0),
            ) for c in sum(self.dataset_cfgs, []))
        self.num_procs = max(
            c.get('eval_cfg', {}).get('evaluator', {}).get(
                'judge_cfg', {}).get('run_cfg', {}).get('num_procs', 1)
            for c in sum(self.dataset_cfgs, []))
        self.dump_details = (cfg.get('eval', {}).get('runner', {}).get(
            'task', {}).get('dump_details', False))
        self.cal_extract_rate = (cfg.get('eval', {}).get('runner', {}).get(
            'task', {}).get('cal_extract_rate', False))

    def get_command(self, cfg_path, template):
        sys.path.append(os.getcwd())
        script_path = __file__
        if self.num_gpus > 1:
            port = random.randint(12000, 32000)
            command = (f'torchrun --master_port={port} '
                       f'--nproc_per_node {self.num_procs} '
                       f'{script_path} {cfg_path}')
        else:
            python = sys.executable
            command = f'{python} {script_path} {cfg_path}'
        return template.format(task_cmd=command)

    def run(self):
        for model_cfg, dataset_cfgs in zip(self.model_cfgs, self.dataset_cfgs):
            for dataset_cfg in dataset_cfgs:
                self.model_cfg = model_cfg
                self.dataset_cfg = dataset_cfg

                # Load Dataset
                self.eval_cfg = copy.deepcopy(dataset_cfg.get('eval_cfg'))
                self.output_column = copy.deepcopy(
                    dataset_cfg['reader_cfg']['output_column'])

                out_path = get_infer_output_path(
                    self.model_cfg,
                    self.dataset_cfg,
                    osp.join(self.work_dir, 'results'),
                )
                if osp.exists(out_path):
                    continue
                self._score()

    def _score(self):
        # Load and preprocess test data
        test_set = self._load_and_preprocess_test_data()
        # Load predictions
        pred_dicts, pred_strs = self._load_predictions()

        # Process predictions
        pred_strs = self._process_predictions(pred_strs)

        # Evaluate predictions
        result = self._evaluate_predictions(
            pred_strs,
            test_set,
            pred_dicts,
        )

        # Save results
        self._save_results(result)

    def _load_and_preprocess_test_data(self):
        """Load test dataset and apply postprocessing if needed."""
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

        return test_set

    def _load_predictions(self):
        """Load model predictions from files."""
        filename = get_infer_output_path(
            self.model_cfg,
            self.dataset_cfg,
            osp.join(self.work_dir, 'predictions'),
        )
        # in case the prediction is partial
        root, ext = osp.splitext(filename)
        partial_filename = root + '_0' + ext

        if not osp.exists(osp.realpath(filename)) and not osp.exists(
                osp.realpath(partial_filename)):
            raise FileNotFoundError(
                f'Prediction files not found: neither {filename} '
                f'nor {partial_filename} exists')

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

        return pred_dicts, pred_strs

    def _process_predictions(self, pred_strs):
        """Apply various processing steps to predictions."""
        # Check if we're dealing with a list of lists (pred_list_flag)
        pred_list_flag = pred_strs is not None and isinstance(
            pred_strs[0], list)

        # Extract role predictions if needed
        if ('pred_role' in self.eval_cfg and 'meta_template' in self.model_cfg
                and not MODELS.get(self.model_cfg['type']).is_api):
            # Create a prompt template for role config parsing
            from opencompass.models.base import LMTemplateParser

            parser = LMTemplateParser(self.model_cfg['meta_template'])
            role = parser.roles[self.eval_cfg['pred_role']]
            if pred_list_flag:
                pred_strs = [[
                    extract_role_pred(
                        _pred,
                        role.get('begin', None),
                        role.get('end', None),
                    ) for _pred in pred
                ] for pred in pred_strs]
            else:
                pred_strs = [
                    extract_role_pred(
                        pred,
                        role.get('begin', None),
                        role.get('end', None),
                    ) for pred in pred_strs
                ]

        # Apply postprocessors if configured
        # Postprocess predictions if necessary
        # Model Specified Postprocessor
        if 'pred_postprocessor' in self.model_cfg:
            kwargs = copy.deepcopy(self.model_cfg['pred_postprocessor'])
            proc = kwargs.pop('type')
            if isinstance(proc, str):
                proc = TEXT_POSTPROCESSORS.get(proc)
            if pred_list_flag:
                pred_strs = [[proc(s, **kwargs) for s in preds]
                             for preds in pred_strs]
            else:
                pred_strs = [proc(s, **kwargs) for s in pred_strs]

        # Dataset Specified Postprocessor
        if 'pred_postprocessor' in self.eval_cfg:
            kwargs = copy.deepcopy(self.eval_cfg['pred_postprocessor'])
            proc = kwargs.pop('type')
            if isinstance(proc, str):
                proc = TEXT_POSTPROCESSORS.get(proc)
            if pred_list_flag:
                pred_strs = [[proc(s, **kwargs) for s in preds]
                             for preds in pred_strs]
            else:
                pred_strs = [proc(s, **kwargs) for s in pred_strs]

        return pred_strs

    def _evaluate_predictions(
        self,
        pred_strs,
        test_set,
        pred_dicts,
    ):
        """Evaluate predictions using the configured evaluator."""
        # Get references from test set
        references = (None if self.output_column is None else
                      [sample[self.output_column] for sample in test_set])
        # Build evaluator from config
        evaluator_cfg = self.eval_cfg.get('evaluator', {})
        evaluator_type = evaluator_cfg.get('type')
        if isinstance(evaluator_type, str):
            evaluator_type = ICL_EVALUATORS.get(evaluator_type)

        # Prepare evaluator inputs
        evaluator_cfg_copy = copy.deepcopy(evaluator_cfg)
        evaluator_cfg_copy.pop('type', None)
        # Initialize evaluator with appropriate parameters
        sig = signature(evaluator_type)
        if 'predictions' in sig.parameters and 'references' in sig.parameters:
            evaluator = evaluator_type(
                predictions=pred_strs,
                references=references,
                **evaluator_cfg_copy,
            )
        else:
            evaluator = evaluator_type(**evaluator_cfg_copy)

        # Set output directory for the evaluator
        out_path = get_infer_output_path(
            self.model_cfg,
            self.dataset_cfg,
            osp.join(self.work_dir, 'results'),
        )
        evaluator._out_dir = osp.splitext(out_path)[0]  # strip extension

        # If preds contains keys that match the score method
        # parameters, include them
        if pred_dicts:
            preds = {
                k: [pred.get(k) for pred in pred_dicts]
                for k in pred_dicts[0]
            }
        # Add predictions and references if they're expected
        # by the score method
        preds['predictions'] = pred_strs
        preds['references'] = (test_set[self.output_column]
                               if self.output_column else None)
        preds['test_set'] = test_set
        if 'origin_prompt' not in preds:
            try:
                preds['origin_prompt'] = [None for _ in range(len(pred_strs))]
            except TypeError:
                preds['origin_prompt'] = None
        preds = {k: preds[k] for k in signature(evaluator.score).parameters}
        # Call evaluate with the appropriate parameters
        k = self.dataset_cfg.get('k', 1)
        n = self.dataset_cfg.get('n', 1)
        result = evaluator.evaluate(k, n, copy.deepcopy(test_set), **preds)

        # Format details if needed
        if self.dump_details:
            # Get detailed results if available
            details = result.get('details', None)
            if details is None:
                self.logger.info(
                    'Details is not give by evaluator, try to format it')
                try:
                    result['details'] = self.format_details(
                        pred_strs,
                        references,
                        details,
                        pred_dicts,
                    )

                    # Calculate extraction rate if needed
                    if self.cal_extract_rate and details is not None:
                        result['extract_rate'] = self.extract_rate(result)

                    # Calculate BPB if applicable
                    if pred_dicts and 'BPB' in pred_dicts[0].get(
                            list(pred_dicts[0].keys())[0], {}):
                        correct_bpb, incorrect_bpb = self.calculate_bpb(
                            pred_dicts)
                        result['correct_bpb'] = correct_bpb
                        result['incorrect_bpb'] = incorrect_bpb
                except Exception as e:
                    self.logger.warning(f'Skip dumping details due to: {e}.')
        else:
            result.pop('details', None)
        return result

    def _save_results(self, result):
        """Save evaluation results to file."""
        out_path = get_infer_output_path(
            self.model_cfg,
            self.dataset_cfg,
            osp.join(self.work_dir, 'results'),
        )
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

    def format_details(
        self,
        predictions,
        references,
        details,
        pred_dicts,
    ):
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
