import json
import os
import os.path as osp
import re
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import mmengine
from datasets import Dataset
from mmengine.config import ConfigDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.registry import (DICT_POSTPROCESSORS, ICL_EVALUATORS,
                                  ICL_PROMPT_TEMPLATES)
from opencompass.utils import build_dataset_from_cfg, build_model_from_cfg
from opencompass.utils.logging import get_logger

logger = get_logger(__name__)


def fix_json_slash(s: str) -> str:
    return re.sub(r'(?<!\\)\\(?![\\/\"bfnrtu])', r'\\\\', s)


def sage_pred_postprocess(
    prediction: str, think_tags: Tuple[str,
                                       str] = ('<think>', '</think>')) -> str:
    if prediction is None:
        prediction = ''
    if think_tags[1] in prediction:
        prediction = prediction.split(think_tags[1])[-1]
    json_str = re.search(r'```json\n(.*?)\n```', prediction, re.DOTALL)
    if json_str is None:
        json_str = re.search(r'```(.*?)```', prediction, re.DOTALL)
        if json_str is None:
            try:
                assert '"answers":' in prediction
                json_content = '{\n"answers"' + prediction.split(
                    '"answers"')[-1]
                json_content = fix_json_slash(json_content)
                return {'extract_error': False, 'answers': json_content}
            except Exception:
                return {'extract_error': True, 'answers': '{"answers": []}'}
        else:
            json_content = json_str.group(
                1).strip() if json_str.lastindex else json_str.group(0)
            json_content = fix_json_slash(json_content)
            return {'extract_error': False, 'answers': json_content}
    else:
        json_content = json_str.group(
            1).strip() if json_str.lastindex else json_str.group(0)
        json_content = fix_json_slash(json_content)
        return {'extract_error': False, 'answers': json_content}


def get_final_results(parsed_judges: List[List[Dict]],
                      references: List[List[str]],
                      origial_judges: List[List[str]]) -> Dict:
    count = 0
    details = []
    for parsed_judge, reference, origial_judge in zip(parsed_judges,
                                                      references,
                                                      origial_judges):
        detail = {
            'origial_judge':
            origial_judge,
            'reference':
            reference,
            'parsed_judge':
            parsed_judge,
            'correct':
            sum([item['overall_judge'] for item in parsed_judge]) >
            (len(parsed_judge) // 2)
        }
        count += 1
        details.append(detail)

    accuracy = sum(detail['correct'] for detail in details) / count
    result = {
        'accuracy': accuracy * 100,
        'details': details,
    }
    return result


def process_judge_output(
    output: Dict, think_tags: Tuple[str, str] = ('<think>', '</think>')
) -> Tuple[List[str], List[Dict], List[str]]:

    def _parse(prediction: str) -> dict:
        if think_tags[1] in prediction:
            prediction = prediction.split(think_tags[1])[-1]

        json_str = re.search(r'```json(.*?)```', prediction, re.DOTALL)
        if json_str is None:
            json_str = re.search(r'```(.*?)```', prediction, re.DOTALL)
        if json_str is not None:
            json_content = json_str.group(1).strip()
            json_content = fix_json_slash(json_content)
            try:
                return json.loads(json_content)
            except Exception:
                return {
                    'judgements': [{
                        'label':
                        'C',
                        'explanation':
                        'Error processing judge output'
                    }]
                }
        else:
            try:
                fallback = '{"judgements"' + prediction.split(
                    '"judgements"')[-1]
                fallback = fix_json_slash(fallback)
                return json.loads(fallback)
            except Exception:
                try:
                    fallback = '{"judgements": [{"label"' + prediction.split(
                        '"label"')[-1] + '}]}'
                    fallback = fix_json_slash(fallback)
                    return json.loads(fallback)
                except Exception:
                    return {
                        'judgements': [{
                            'label':
                            'C',
                            'explanation':
                            'Error processing judge output'
                        }]
                    }

    origial_judges = []
    parsed_judges = []
    references = []
    for k, v in output.items():
        parsed_judge = _parse(v['prediction'])
        try:
            if isinstance(parsed_judge, list):
                judgements = parsed_judge
            else:
                judgements = parsed_judge['judgements']
        except Exception:
            judgements = [{
                'label': 'C',
                'explanation': 'Error processing judge output'
            }]

        judgements = judgements if isinstance(judgements,
                                              list) else [judgements]
        all_judge_labels = [item['label'] == 'A' for item in judgements]
        origial_judges.append(v['prediction'])
        parsed_judges.append({
            'judgements': judgements,
            'overall_judge': all(all_judge_labels)
        })
        references.append(v['gold'])

    return origial_judges, parsed_judges, references


def sage_judge_postprocess(
    output: List[Dict],
    output_path: str,
    think_tags: Tuple[str, str] = ('<think>', '</think>')
) -> dict:
    origial_judges_list, parsed_judges_list, references_list = [], [], []
    for _output in output.values():
        origial_judges, parsed_judges, references = process_judge_output(
            _output, think_tags)
        origial_judges_list.append(origial_judges)
        parsed_judges_list.append(parsed_judges)
        references_list.append(references)
    origial_judges_list = [[item[i] for item in origial_judges_list]
                           for i in range(len(origial_judges_list[0]))]
    references_list = [[item[i] for item in references_list]
                       for i in range(len(references_list[0]))]
    parsed_judges_list = [[item[i] for item in parsed_judges_list]
                          for i in range(len(parsed_judges_list[0]))]
    results = get_final_results(parsed_judges_list, references_list,
                                origial_judges_list)
    return results


@ICL_EVALUATORS.register_module()
class SAGELLMEvaluator(BaseEvaluator):
    """Generic LLM evaluator using majority voting.

    Arguments:
        prompt_template (ConfigDict): The prompt template for evaluation.
        judge_cfg (list of ConfigDict): A list of config for Judge LLM.
        dataset_cfg (ConfigDict): The config for dataset.
        pred_postprocessor (ConfigDict): The config for postprocessor.
            used for the prediction results.
        dict_postprocessor (ConfigDict): The config for postprocessor,
            used for evaluation results dict.
    """

    def __init__(
        self,
        prompt_template: ConfigDict,
        judge_cfg: List[ConfigDict],
        dataset_cfg: Optional[ConfigDict] = None,
        pred_postprocessor: Optional[ConfigDict] = None,
        dict_postprocessor: Optional[ConfigDict] = None,
        keep_predictions: bool = False,
    ) -> None:
        super().__init__(pred_postprocessor=pred_postprocessor)
        if not judge_cfg:
            self.judge_cfg = [self.default_judge_cfg]
        else:
            assert isinstance(judge_cfg.get('judgers', None),
                              List), 'judge_cfg must be a list'
            self.judge_cfg = judge_cfg.get('judgers')
        self.output_path = ''

        self.prompt_template = ICL_PROMPT_TEMPLATES.build(prompt_template)

        # Build Dataset
        self.dataset_cfg = dataset_cfg
        assert dataset_cfg is not None, 'dataset_cfg is None'

        self.dict_postprocessor = dict_postprocessor
        self.pred_postprocessor = pred_postprocessor

    def build_inferencer(self):
        """Build LLM Inference."""

        # Build LLM Inference
        self.inferencer = []
        for _judge_cfg in self.judge_cfg:
            output_path = f'{self._out_dir}_replica{self.dataset_replica_idx}_{_judge_cfg["path"].split("/")[-1]}.json'  # noqa
            logger.info(f'LLM judge details will be saved at:{output_path}')
            out_dir, out_name = osp.split(output_path)

            logger.info(
                f'Set self.output_path to {output_path} for current task')
            assert output_path is not None, 'output_path is None'

            max_out_len = _judge_cfg.get('max_out_len', None)
            batch_size = _judge_cfg.get('batch_size', None)

            model = build_model_from_cfg(model_cfg=_judge_cfg)

            self.inferencer.append(
                GenInferencer(
                    model,
                    max_out_len=max_out_len,
                    batch_size=batch_size,
                    output_json_filepath=out_dir,
                    output_json_filename=out_name,
                ))

    def score(
        self,
        predictions,
        references: Optional[List] = None,
        test_set: Optional[Dataset] = None,
    ) -> Dict:
        """Apply to single-model scoring.

        Args:
            predictions: List of model predictions
            references: List of reference answers
            test_set: Optional Dataset containing additional
            context for evaluation
        """
        assert len(predictions) == len(
            references), 'predictions and references must have the same length'

        # -------------- Build Inferencer ----------------
        self.build_inferencer()
        # ---------------- Process Predictions ------------------
        answers = [item['answers'] for item in predictions]

        # For Single Round Dialogue
        prediction_dict = {'prediction': answers, 'obj_gold': references}

        # ---------------- Build Dataset for LLM Judge -----------------
        if self.dataset_cfg:
            dataset = build_dataset_from_cfg(self.dataset_cfg)
            for k, v in prediction_dict.items():
                dataset.reader.dataset['test'] = dataset.test.add_column(k, v)
                dataset.reader.input_columns.append(k)

            if references:
                dataset.reader.input_columns.append('reference')
                dataset.reader.dataset['test'] = dataset.test.add_column(
                    'reference', references)
        else:
            # Handle test_set in the else branch
            from opencompass.datasets.lmeval import LMEvalDataset

            if test_set is not None:
                # If test_set is provided, use it as the base
                # Ensure necessary columns exist
                if 'prediction' not in test_set.column_names:
                    test_set = test_set.add_column('prediction', answers)
                if 'reference' not in test_set.column_names:
                    test_set = test_set.add_column('reference', references)

                # Prepare input_columns and data dictionary
                input_columns = test_set.column_names
                data_dict = {
                    column: test_set[column]
                    for column in test_set.column_names
                }
            else:
                # Original default dataset building logic
                input_columns = list(prediction_dict.keys())
                if references:
                    input_columns.append('reference')
                data_dict = prediction_dict.copy()
                if references:
                    data_dict['reference'] = references

            # Create LMEvalDataset
            dataset = LMEvalDataset(
                reader_cfg=dict(
                    input_columns=input_columns,
                    output_column=None,
                    train_split='test',
                ),
                **data_dict,
            )

        if osp.exists(f'{self._out_dir}'
                      f'_replica{self.dataset_replica_idx}'
                      f'_combine.json'):
            output = mmengine.load(f'{self._out_dir}'
                                   f'_replica{self.dataset_replica_idx}'
                                   f'_combine.json')
        else:
            dataset.reader.output_column = 'reference'
            retriever = ZeroRetriever(dataset)
            output = {}
            # ----------------- LLM Judge ----------------
            for inferencer in self.inferencer:
                key = inferencer.output_json_filename.split('.')[0].split(
                    '_')[-1]
                if osp.exists(
                        osp.join(inferencer.output_json_filepath,
                                 inferencer.output_json_filename)):
                    output[key] = mmengine.load(
                        osp.join(inferencer.output_json_filepath,
                                 inferencer.output_json_filename))
                else:
                    inferencer.inference(retriever=retriever,
                                         prompt_template=self.prompt_template)
                    output[key] = mmengine.load(
                        osp.join(inferencer.output_json_filepath,
                                 inferencer.output_json_filename))

            mmengine.dump(output, f'{self._out_dir}'
                          f'_replica{self.dataset_replica_idx}'
                          f'_combine.json',
                          indent=4)
        return self.output_postprocess(output, dataset)

    def output_postprocess(self, output: Dict, dataset=None) -> Dict:
        """Postprocess output by adding necessary statistics or data into
        it."""
        import inspect

        if self.dict_postprocessor is None:
            return output
        else:
            kwargs = deepcopy(self.dict_postprocessor)
            proc = DICT_POSTPROCESSORS.get(kwargs.pop('type'))
            sig = inspect.signature(proc)
            if 'dataset' in sig.parameters:
                return proc(output,
                            self.output_path,
                            dataset=dataset,
                            **kwargs)
            else:
                return proc(output, self.output_path, **kwargs)

    @property
    def default_judge_cfg(self):
        from opencompass.models import OpenAISDK
        logger.info('Please set your judge model in `OC_JUDGE_MODEL`, \
            `OC_JUDGE_API_KEY`, `OC_JUDGE_API_BASE` environment variables.')
        DEFAULT_JUDGE_CFG = dict(
            type=OpenAISDK,
            path=os.environ['OC_JUDGE_MODEL'],
            key=os.environ['OC_JUDGE_API_KEY'],
            openai_api_base=[
                os.environ.get('OC_JUDGE_API_BASE',
                               'https://api.openai.com/v1/')
            ],
            meta_template=dict(round=[
                dict(role='HUMAN', api_role='HUMAN'),
                dict(role='BOT', api_role='BOT', generate=True),
            ], ),
            query_per_second=16,
            batch_size=1024,
            temperature=0.001,
            tokenizer_path='gpt-4o-2024-05-13',
            verbose=True,
            max_out_len=16384,
            max_seq_len=49152,
        )

        return DEFAULT_JUDGE_CFG
