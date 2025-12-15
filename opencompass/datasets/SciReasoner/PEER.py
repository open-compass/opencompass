# flake8: noqa
# dataset: PEER
# task : solubility prediction

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union

import numpy as np
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from openai import OpenAI
from sklearn.metrics import f1_score, precision_score, recall_score

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path


@LOAD_DATASET.register_module()
class PEER_Dataset(BaseDataset):

    @staticmethod
    def load(path, task, max_cut=-1, mini_set=False, hf_hub=False):

        # if (hf_hub is True):
        #     # load from huggingface hub
        #     train_data = []
        #     repo_id = test_path.split('/')[0] + '/' + test_path.split('/')[1]
        #     train_path = train_path.split(repo_id + '/')[1]
        #     test_path = test_path.split(repo_id + '/')[1]
        #
        #     train_path = hf_hub_download(repo_id,
        #                                  train_path,
        #                                  repo_type='dataset')
        #     test_path = hf_hub_download(repo_id,
        #                                 test_path,
        #                                 repo_type='dataset')

        path = get_data_path(path)
        train_path = os.path.join(path, f'{task}/dev/data.json')
        test_path = os.path.join(path, f'{task}/test/data.json')

        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        train_data = train_data[:5]
        # Limit the dataset to 5 samples for testing purposes

        if (max_cut != -1):
            test_data = test_data[:max_cut]
        if mini_set:
            import random
            random.seed(1024)
            test_data = random.sample(test_data, 150)
            random.seed()

        dataset = DatasetDict({
            'train': Dataset.from_list(train_data),
            'test': Dataset.from_list(test_data)
        })
        return dataset


@TEXT_POSTPROCESSORS.register_module()
def PEER_postprocess_default(text: Union[str, None]) -> str:
    text = text.strip()
    text = re.sub(r'<\|endoftext\|>', '', text)
    text = re.sub(r'<\|im_end\|>', '', text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text


@TEXT_POSTPROCESSORS.register_module()
def PEER_postprocess(text: Union[str, None]) -> str:
    """
        从模型的原始输出中提取预测结果（Yes或No）。

        此函数会查找并返回跟在The answer is后面的Yes或者No，
        或从文本中识别常见的Yes/No表达方式。
        """
    # 检查输入是否为字符串，提高代码健壮性
    if not isinstance(text, str):
        return ''
    # 定义正则表达式模式，匹配常见的Yes/No表达方式
    # 首先检查是否有明确的"The answer is Yes/No"模式
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'.*?</think>\s*', '', text, flags=re.DOTALL)
    match = re.search(r'The answer is\s+(Yes|No)', text, re.IGNORECASE)
    if match:
        return match.group(1)

    # 检查常见的肯定表达方式
    positive_patterns = [
        r'will be soluble',
        r'will dissolve',
        r'is soluble',
        r'can be predicted',
        r'positive',
        r'Yes',
        r'correct',
        r'valid',
        r'accurate',
        r'certainly',
        r'indeed',
        r'affirmative',
        r'highly soluble',
        r'easily soluble',
        r'dissolves easily',
        r'is assured',
        # r'likely',
        r'be soluble'
    ]

    # 检查常见的否定表达方式
    negative_patterns = [
        r'will not be soluble',
        r'is not soluble',
        r'will not dissolve',
        r'low solubility',
        r'low',
        r'cannot be predicted',
        r'negative',
        r'No',
        r'incorrect',
        r'invalid',
        r'inaccurate',
        r'impossible',
        r'not possible',
        r'denied',
        r'be insoluble',
    ]

    # 检查是否包含肯定表达
    for pattern in positive_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return 'Yes'

    # 检查是否包含否定表达
    for pattern in negative_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return 'No'

    # 若无法识别，返回空字符串
    return ''


@TEXT_POSTPROCESSORS.register_module()
def PEER_postprocess_float_compare(text: Union[str, None],
                                   compare_number: float) -> str:
    # 从模型的输出中匹配预测的数值，与compare_number进行比较, 大于则返回"Yes"，否则返回"No"
    if not isinstance(text, str):
        return ''
    try:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'.*?</think>\s*', '', text, flags=re.DOTALL)
        # 提取文本中的数字
        match = re.search(r'[-+]?\d*\.\d+|\d+', text)
        if match:
            value = float(match.group(0))
            # 比较数值
            if value > compare_number:
                return 'Yes'
            else:
                return 'No'
        else:
            # 如果没有找到数字，返回空字符串
            return ''
    except ValueError:
        # 如果转换失败，返回空字符串
        return ''


def calculate_accuracy(pred_text_list, gold_text_list):
    assert len(pred_text_list) == len(gold_text_list)
    num_all = len(pred_text_list)
    metrics = {}
    metrics['num_all'] = num_all
    num_no_answer = 0
    num_invalid = 0
    num_correct = 0
    new_pred_text_list, new_gold_text_list = [], []
    for (pred_item, gold_item) in zip(pred_text_list, gold_text_list):
        if pred_item is None or pred_item == '':
            num_no_answer += 1
            continue
        assert len(pred_item) == 1
        assert len(gold_item) == 1
        pred_item = pred_item[0].strip().lower()
        gold_item = gold_item[0].strip().lower()
        if pred_item == '':
            num_no_answer += 1
            continue
        if pred_item not in ('yes', 'no'):
            num_invalid += 1
            continue
        pred_item = 1 if pred_item == 'yes' else 0
        gold_item = 1 if gold_item == 'yes' else 0
        new_pred_text_list.append(pred_item)
        new_gold_text_list.append(gold_item)
        if gold_item == pred_item:
            num_correct += 1

    metrics['num_no_answer'] = num_no_answer
    metrics['num_invalid'] = num_invalid
    metrics['num_correct'] = num_correct

    # return metrics

    new_gold_text_list = np.array(new_gold_text_list)
    new_pred_text_list = np.array(new_pred_text_list)

    # macro_roc_auc_score =
    # roc_auc_score(new_gold_text_list, new_pred_text_list)
    f1 = f1_score(new_gold_text_list, new_pred_text_list)
    # metrics['roc_auc_score'] = macro_roc_auc_score
    metrics['accuracy'] = num_correct / (num_all) * 100
    metrics['acc_wo_no_answer_invalid'] = num_correct / (
        num_all - num_no_answer - num_invalid) * 100 if (
            num_all - num_no_answer - num_invalid) > 0 else 0
    metrics['precision'] = precision_score(new_gold_text_list,
                                           new_pred_text_list) * 100
    metrics['recall'] = recall_score(new_gold_text_list,
                                     new_pred_text_list) * 100
    metrics['f1_score'] = f1 * 100

    return metrics


# ----------------------------------------------------------------------
# 定义 Evaluator (评估器) - 这是修改的核心
# ----------------------------------------------------------------------

MAX_RETRIES = 3
BACKOFF_SEC = 2


class PEER_Evaluator(BaseEvaluator):

    def __init__(self,
                 task='solubility',
                 gpt_model='gpt-4',
                 openai_key='xxx',
                 use_gpt=True,
                 max_workers=8,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task
        self.gpt_model = gpt_model
        self.use_gpt = use_gpt
        self.max_workers = max_workers

        if task in [
                'stability',
        ]:
            self.use_gpt = False

        if self.use_gpt:
            if not openai_key:
                raise ValueError('OpenAI API key is missing.')
            self.client = OpenAI(base_url='url', api_key=openai_key)

    def _retry_api(self, fn, *args, **kwargs):
        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = fn(*args, **kwargs)
                if result is not None:
                    return result
                raise ValueError('Received None')
            except Exception as e:
                last_exc = e
                sleep_time = BACKOFF_SEC**attempt
                print(f'[retry] attempt {attempt} failed ({e}),'
                      f' retrying in {sleep_time}s…')
                time.sleep(sleep_time)
        raise last_exc

    def ask_gpt25(self, question, answer, prediction):

        prompt = (
            'Please determine whether this answer is correct. Definition:'
            "'Correct': The core conclusion of the model's answer (if any) is "
            'completely consistent with the reference answer (literal identity'
            " is not required). 'Incorrect': The core conclusion of the"
            " model's answer is consistent with the reference answer, or the"
            ' core conclusion is not clearly expressed. Reference answer'
            f': {answer}'
            f'Model answer: {prediction}'
            "If correct, answer 'True'; if incorrect, answer 'False'."
            "Please only answer 'True' or 'False'.")

        def _call():
            response = self.client.chat.completions.create(
                model=self.gpt_model,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                temperature=0)

            result = response.choices[0].message.content.strip().upper()
            print('=== GPT 判断结果 ===')
            print(f'Prompt:\n{prompt}')
            print(f'Output:\n{result}')
            return result

        try:
            return self._retry_api(_call)
        except Exception as e:
            print(f'[GPT ERROR] Exception: {e}')
            return ''

    def ask_gpt25_batch(self, questions, answers, predictions):
        results = [None] * len(questions)

        def task(index, q, a, p):
            try:
                result = self.ask_gpt25(q, a, p)
                results[index] = result
            except Exception as e:
                results[index] = ''
                print(f'[GPT ERROR] 批次样本 {index} 出错: {e}')

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(task, i, q, a, p)
                for i, (q, a,
                        p) in enumerate(zip(questions, answers, predictions))
            ]
            for future in as_completed(futures):
                pass

        return results

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        if not isinstance(predictions[0], list):
            predictions = [[pred] for pred in predictions]
        if not isinstance(references[0], list):
            references = [[ref] for ref in references]

        postprocessed_references = [[PEER_postprocess(r[0]).strip().lower()]
                                    for r in references]
        postprocessed_predictions = [[PEER_postprocess(p[0]).strip().lower()]
                                     for p in predictions]

        voted_prediction = []
        for pred in postprocessed_predictions:
            valid_pred = [p for p in pred if p in ['yes', 'no']]
            cnt = valid_pred.count('yes')
            if cnt > len(valid_pred) / 2:
                voted = 'yes'
            elif cnt < len(valid_pred) / 2:
                voted = 'no'
            else:
                voted = ''
            voted_prediction.append([voted])

        num_all = len(voted_prediction)
        num_correct, num_no_answer, num_invalid = 0, 0, 0
        num_gpt_called = 0
        new_pred, new_gold = [], []

        to_recheck_indices = []
        to_recheck_golds = []
        to_recheck_preds = []

        for i, (pred_item, gold_item) in enumerate(
                zip(postprocessed_predictions, postprocessed_references)):
            pred = pred_item[0]
            gold = gold_item[0]

            if pred not in ('yes', 'no'):
                to_recheck_indices.append(i)
                to_recheck_golds.append(references[i][0])
                to_recheck_preds.append(predictions[i][0])
                continue

            if pred == 'yes':
                pred_bin = 1
            elif pred == 'no':
                pred_bin = 0
            else:
                to_recheck_indices.append(i)
                to_recheck_golds.append(references[i][0])
                to_recheck_preds.append(predictions[i][0])
                continue

            if gold == 'yes':
                gold_bin = 1
            elif gold == 'no':
                gold_bin = 0
            else:
                to_recheck_indices.append(i)
                to_recheck_golds.append(references[i][0])
                to_recheck_preds.append(predictions[i][0])
                continue

            if pred_bin == gold_bin:
                num_correct += 1
                # import pdb; pdb.set_trace()
                print(references[i][0], '\n', predictions[i][0], '----')
                new_pred.append(pred_bin)
                new_gold.append(gold_bin)
            else:
                to_recheck_indices.append(i)
                to_recheck_golds.append(references[i][0])
                to_recheck_preds.append(predictions[i][0])

        if to_recheck_indices and self.use_gpt:
            rechecked_preds = self.ask_gpt25_batch(
                ['' for _ in to_recheck_indices], to_recheck_golds,
                to_recheck_preds)
            num_gpt_called += len(rechecked_preds)

            for i, result in enumerate(rechecked_preds):
                result = result.strip().lower()
                if 'true' in result:
                    num_correct += 1
                    pred_bin = 1
                    gold_bin = 1
                elif 'false' in result:
                    pred_bin = 0
                    gold_bin = 1
                else:
                    pred_bin = 1
                    gold_bin = 0

                new_pred.append(pred_bin)
                new_gold.append(gold_bin)

        new_pred = np.array(new_pred)
        new_gold = np.array(new_gold)

        metrics = {
            'num_all':
            num_all,
            'num_correct':
            num_correct,
            'num_no_answer':
            num_no_answer,
            'num_invalid':
            num_invalid,
            'num_gpt_called':
            num_gpt_called,
            'accuracy':
            num_correct / num_all * 100,
            'acc_wo_no_answer_invalid':
            num_correct / (num_all - num_no_answer - num_invalid) * 100 if
            (num_all - num_no_answer - num_invalid) > 0 else 0,
            'precision':
            precision_score(new_gold, new_pred, zero_division=0) * 100,
            'recall':
            recall_score(new_gold, new_pred, zero_division=0) * 100,
            'f1_score':
            f1_score(new_gold, new_pred, zero_division=0) * 100,
        }

        return metrics
