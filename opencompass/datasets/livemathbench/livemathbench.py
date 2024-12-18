import concurrent.futures
import os
import re
from collections import OrderedDict
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List

import jsonlines
import numpy as np
from datasets import Dataset

from opencompass.models import OpenAISDK
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET, MODELS
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .prompts import (EXTRACT_PROMPT_CN, EXTRACT_PROMPT_EN, JUDGE_PROMPT_CN,
                      JUDGE_PROMPT_EN, PROMPT_CN, PROMPT_EN)


@LOAD_DATASET.register_module()
class LiveMathBenchDataset(BaseDataset):

    @staticmethod
    def load(
        path: str,
        k: int,
        n: int,
        dataset_splits: List[str] = [
            'AIMC', 'CEE', 'CMO', 'MATH500', 'AIME2024'
        ],
        dataset_languages: List[str] = ['cn', 'en'],
    ) -> List[Dict[str, Any]]:
        dataset = []
        dataset_info = {}
        path = get_data_path(path)
        for split, language in product(dataset_splits, dataset_languages):
            file_path = os.path.join(path, f'{split}_{language}.jsonl')
            if not os.path.exists(file_path):
                continue
            dataset_info[f'{split}_{language}'] = {
                'single-choice': 0,
                'multiple-choice': 0,
                'fill-in-the-blank': 0,
                'problem-solving': 0
            }
            question_type_mapping = {
                '单选': 'single-choice',
                '多选': 'multiple-choice',
                '填空': 'fill-in-the-blank',
                '问答': 'problem-solving'
            }
            with jsonlines.open(file_path, 'r') as file:
                for example_idx, example in enumerate(file):
                    dataset_info[f'{split}_{language}'][
                        example['question_type'] if language == 'en' else
                        question_type_mapping[example['question_type']]] += 1

                    prompt = PROMPT_EN if language == 'en' else PROMPT_CN
                    example.update({
                        'dataset_key':
                        f'{split}_{language}_{example_idx}',
                        'prompt':
                        prompt.format(question_type=example['question_type'],
                                      question=example['question'] +
                                      ('' if 'options' not in example else
                                       ' '.join(example['options']))),
                        'k':
                        k,
                        'n':
                        n
                    })
                    for idx in range(k * n):
                        duplicated_example = deepcopy(example)
                        duplicated_example.update({'duplicated_idx': idx})
                        dataset.append(duplicated_example)

        return Dataset.from_list(dataset)


@ICL_EVALUATORS.register_module()
class LiveMathBenchEvaluator(BaseEvaluator):
    api_meta_template = dict(round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ])

    def __init__(self,
                 model_name,
                 url,
                 with_postprocess=True,
                 use_extract_model=False,
                 post_url=[],
                 post_model_name='',
                 **kwargs):
        if isinstance(url, str):
            url = [url]

        self.model = [
            MODELS.build(
                dict(
                    type=OpenAISDK,
                    path=model_name,
                    openai_api_base=url,
                    key='EMPTY',
                    query_per_second=128,
                    meta_template=self.api_meta_template,
                    temperature=kwargs.get('temperature', 0.001),
                    max_seq_len=kwargs.get('max_tokens', 16384),
                )) for url in url
        ]
        self.with_postprocess = with_postprocess
        self.use_extract_model = use_extract_model
        self.post_url = post_url
        self.post_model_name = post_model_name

    def batch_response(self, models: List[OpenAISDK],
                       inputs: List[str]) -> List[str]:
        batch_num = len(models)
        batch_size = (len(inputs) + batch_num - 1) // batch_num
        result_responses = []

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=batch_num) as executor:
            futures = [
                executor.submit(models[i].generate,
                                inputs[i * batch_size:(i + 1) * batch_size])
                for i in range(batch_num)
            ]
            for response in executor.map(lambda f: f.result(), futures):
                result_responses.extend(response)

        return result_responses

    def postprocess(self, questions: List[str], predictions: List[str],
                    question_types: List[str],
                    languages: List[str]) -> List[str]:
        if self.use_extract_model:
            assert len(self.post_url) > 0 and self.post_model_name != ''
            post_model = [
                MODELS.build(
                    dict(
                        type=OpenAISDK,
                        path=self.post_model_name,
                        openai_api_base=url,
                        key='EMPTY',
                        query_per_second=2,
                        meta_template=self.api_meta_template,
                        temperature=0.01,
                        max_seq_len=1024,
                    )) for url in self.post_url
            ]

            input_prompts = []
            for question, prediction, question_type, language in zip(
                    questions, predictions, question_types, languages):
                prompt = (EXTRACT_PROMPT_EN
                          if language == 'en' else EXTRACT_PROMPT_CN)
                input_prompts.append(
                    prompt.format(question=question,
                                  response=prediction,
                                  question_type=question_type))

            result_responses = self.batch_response(post_model, input_prompts)

            return result_responses

        def last_boxed_only_string(string):
            idx = string.rfind('\\boxed')
            if idx < 0:
                idx = string.rfind('\\fbox')
                if idx < 0:
                    return None

            i = idx
            right_brace_idx = None
            num_left_braces_open = 0
            while i < len(string):
                if string[i] == '{':
                    num_left_braces_open += 1
                if string[i] == '}':
                    num_left_braces_open -= 1
                    if num_left_braces_open == 0:
                        right_brace_idx = i
                        break
                i += 1

            if right_brace_idx is None:
                retval = None
            else:
                retval = string[idx:right_brace_idx + 1]

            return retval

        def remove_boxed(s):
            left = '\\boxed{'
            try:
                assert s[:len(left)] == left
                assert s[-1] == '}'
                return s[len(left):-1]
            except Exception:
                return None

        def extract_boxed_answer(pred_str, strip_double_curly_brace=False):
            boxed_str = last_boxed_only_string(pred_str)
            if boxed_str is None:
                return None
            answer = remove_boxed(boxed_str)
            if answer is None:
                return None
            if strip_double_curly_brace:
                match = re.match('^\{(.*)\}$', answer)  # noqa: W605
                if match:
                    answer = match.group(1)
            return answer

        predictions = [
            extract_boxed_answer(prediction) for prediction in predictions
        ]
        return predictions

    def extract_boxed_answer(self, text):
        match = re.findall(r'\\boxed{(.+?)}', text)
        if match:
            return match[-1]

        return None

    def score(self, predictions, references, origin_prompt, test_set):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        questions = test_set['question']
        question_types = test_set['question_type']
        languages = [key.split('_')[1] for key in test_set['dataset_key']]

        if self.with_postprocess:
            predictions = self.postprocess(questions, predictions,
                                           question_types, languages)

        inputs = []
        for prediction, reference, question, language in zip(
                predictions, references, questions, languages):
            prompt = JUDGE_PROMPT_EN if language == 'en' else JUDGE_PROMPT_CN
            inputs.append(
                prompt.format(answer=prediction,
                              gold_answer=reference,
                              question=question))
        result_responses = self.batch_response(self.model, inputs)
        results = [
            self.extract_boxed_answer(result) == 'yes'
            for result in result_responses
        ]

        K = test_set['k'][0]
        N = test_set['n'][0]
        key2example = {}

        for example, result_response, result, prediction in zip(
                test_set, result_responses, results, predictions):
            if example['dataset_key'] not in key2example:
                key2example[example['dataset_key']] = []
            example.update({
                'eval_response': result_response,
                'prediction': prediction,
                'correct': result
            })
            key2example[example['dataset_key']].append(example)
        for key in key2example:
            key2example[key] = [
                key2example[key][i * K:(i + 1) * K] for i in range(N)
            ]

        count = []
        total_pass_num = []
        details = []
        all_dataset = set()
        for key, examples in key2example.items():
            detail = OrderedDict()
            detail['question'] = examples[0][0]['question']
            detail['answer'] = examples[0][0]['answer']
            detail['responses'] = []
            detail['dataset'] = '_'.join(key.split('_')[:-1])
            all_dataset.add('_'.join(key.split('_')[:-1]))
            if_pass_list = []
            for single_run_examples in examples:
                detail['responses'].append([])
                if_pass_list.append([])
                for example in single_run_examples:
                    detail['responses'][-1].append({
                        'prediction':
                        example['prediction'],
                        'eval_response':
                        example['eval_response']
                    })
                    if_pass_list[-1].append(1.0 if example['correct'] else 0.0)

            if_pass_list = [
                sorted(if_pass, reverse=True) for if_pass in if_pass_list
            ]
            if_pass_list = np.array(if_pass_list)
            i = 1
            while i <= K:
                detail.update({
                    f'pass-rate@{i}':
                    if_pass_list[:, :i].mean(axis=1).mean(axis=0).item(),
                    f'pass-rate@{i}/std':
                    if_pass_list[:, :i].mean(axis=1).std(axis=0).item(),
                    f'pass@{i}':
                    np.ceil(
                        if_pass_list[:, :i].mean(axis=1)).mean(axis=0).item(),
                    f'pass@{i}/std':
                    np.ceil(
                        if_pass_list[:, :i].mean(axis=1)).std(axis=0).item(),
                })
                i = i * 2

            for threshold in [0.5, 0.75, 1.0]:
                detail.update({
                    f'{K}-pass@{threshold}':
                    np.floor(
                        np.where(
                            if_pass_list.mean(axis=1) >= threshold, 1.0,
                            0.0).mean(axis=0))
                })

            count.append(np.ones_like(if_pass_list).sum(axis=1))
            total_pass_num.append(if_pass_list.sum(axis=1))

            details.append(detail)

        detailed_result = OrderedDict()
        detailed_result['details'] = details

        i = 1
        while i <= K:
            detailed_result.update({
                f'pass-rate@{i}':
                100. *
                np.mean([detail[f'pass-rate@{i}'] for detail in details]),
                f'pass-rate@{i}/std':
                100. *
                np.mean([detail[f'pass-rate@{i}/std'] for detail in details]),
                f'pass@{i}':
                100. * np.mean([detail[f'pass@{i}'] for detail in details]),
                f'pass@{i}/std':
                100. * np.mean([detail[f'pass@{i}/std'] for detail in details])
            })
            for d in sorted(list(all_dataset)):
                detailed_result.update({
                    f'{d}/pass-rate@{i}':
                    100. * np.mean([
                        detail[f'pass-rate@{i}']
                        for detail in details if detail['dataset'] == d
                    ]),
                    f'{d}/pass-rate@{i}/std':
                    100. * np.mean([
                        detail[f'pass-rate@{i}/std']
                        for detail in details if detail['dataset'] == d
                    ]),
                    f'{d}/pass@{i}':
                    100. * np.mean([
                        detail[f'pass@{i}']
                        for detail in details if detail['dataset'] == d
                    ]),
                    f'{d}/pass@{i}/std':
                    100. * np.mean([
                        detail[f'pass@{i}/std']
                        for detail in details if detail['dataset'] == d
                    ])
                })
            i = i * 2

            for threshold in [0.5, 0.75, 1.0]:
                detailed_result.update({
                    f'{K}-pass@{threshold}':
                    100. * np.mean([
                        detail[f'{K}-pass@{threshold}'] for detail in details
                    ])
                })
                detailed_result.update({
                    f'{K}-pass@{threshold}/std':
                    100. * np.mean([
                        detail[f'{K}-pass@{threshold}'] for detail in details
                    ])
                })
            for d in sorted(list(all_dataset)):

                for threshold in [0.5, 0.75, 1.0]:
                    detailed_result.update({
                        f'{d}/{K}-pass@{threshold}':
                        100. * np.mean([
                            detail[f'{K}-pass@{threshold}']
                            for detail in details if detail['dataset'] == d
                        ])
                    })
                    detailed_result.update({
                        f'{d}/{K}-pass@{threshold}/std':
                        100. * np.mean([
                            detail[f'{K}-pass@{threshold}']
                            for detail in details if detail['dataset'] == d
                        ])
                    })

        return detailed_result
