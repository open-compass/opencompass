# flake8: noqa

import json
import os
import re
from typing import Union

import numpy as np
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             roc_auc_score)

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path


@LOAD_DATASET.register_module()
class LLM4MatDataset(BaseDataset):

    @staticmethod
    def load(path,
             property,
             train_path,
             test_path,
             mini_set=False) -> DatasetDict:

        def load_single_dataset(path, property, num=None):

            # if (hf_hub is True):
            #     repo_id = path.split('/')[0] + '/' + path.split('/')[1]
            #     path = path.split(repo_id + '/')[1]
            #
            #     path = hf_hub_download(repo_id, path, repo_type='dataset')

            with open(path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            if isinstance(raw_data, dict):
                raw_data = [raw_data]

            processed = []
            for i, item in enumerate(raw_data):
                if not '{' + f'{property} :' in item['output']:
                    continue
                new_item = {
                    'input': item['input'],
                    'output': item['output'],
                }
                processed.append(new_item)
            if num:
                dataset = Dataset.from_list(processed[:num])
            else:
                dataset = Dataset.from_list(processed)
            return dataset

        path = get_data_path(path)
        train_path = os.path.join(path, train_path)
        test_path = os.path.join(path, test_path)

        if mini_set:
            test_num = 150
        else:
            test_num = None
        dataset = DatasetDict({
            'train':
            load_single_dataset(train_path, property, num=5),
            'test':
            load_single_dataset(test_path, property, num=test_num)
        })
        return dataset


non_numeric_props_options = {
    'Direct_or_indirect': ['Indirect', 'Direct'],
    'Direct_or_indirect_HSE': ['Indirect', 'Direct'],
    'SOC': [True, False],
    'is_gap_direct': [True, False],
    'is_stable': [True, False],
}


def remove_think_tags(text: str) -> str:
    if '<think>' not in text:
        return text
    if '</think>' not in text:
        return ''
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)


def extract_strict_value(text: str, property: str) -> str:
    text_clean = re.sub(r'^```(?:json)?\s*|\s*```$',
                        '',
                        text.strip(),
                        flags=re.IGNORECASE | re.MULTILINE)
    try:
        data = json.loads(text_clean)
        if property in data:
            raw_value = data[property]
            if isinstance(raw_value, (int, float)):
                return float(raw_value)
            if property in non_numeric_props_options:
                options = non_numeric_props_options[property]
                for opt in options:
                    if isinstance(opt, bool):
                        if str(raw_value).lower() == str(opt).lower():
                            return str(opt)
                    elif str(raw_value).lower() == str(opt).lower():
                        return opt
                return ''
            return str(raw_value)
    except Exception:
        pass

    pattern = rf'\{{[^{{}}]*"?{re.escape(property)}"?\s*:\s*(.*?)\s*\}}'
    match = re.search(pattern, text_clean, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return ''
    raw_value = match.group(1).strip().strip('"')
    if property in non_numeric_props_options:
        options = non_numeric_props_options[property]
        for opt in options:
            if isinstance(opt, bool):
                if raw_value.lower() == str(opt).lower():
                    return str(opt)
            elif raw_value.lower() == opt.lower():
                return opt
        return ''
    try:
        return float(raw_value)
    except ValueError:
        return ''


@TEXT_POSTPROCESSORS.register_module()
def LLM4Mat_postprocessor(text: Union[str, None], property):
    if text is None or not isinstance(text, str):
        return ''
    text = text.strip()
    text = remove_think_tags(text)
    if text == '':
        return ''
    result = extract_strict_value(text, property)
    return result


class LLM4Mat_Evaluator(BaseEvaluator):

    def score(self, predictions, references):
        is_regression = isinstance(
            references[0],
            (int, float)) and not isinstance(references[0], bool)

        if is_regression:
            y_true = []
            y_pred = []
            total = len(references)
            for t, p in zip(references, predictions):
                try:
                    t_val = float(t)
                    p_val = float(p)
                    if not (np.isfinite(t_val) and np.isfinite(p_val)):
                        continue
                    y_true.append(t_val)
                    y_pred.append(p_val)
                except Exception:
                    continue
            if len(y_true) == 0:
                return {
                    'total': total,
                    'filtered': len(y_true),
                    'MAE': None,
                    'RMSE': None,
                    'MAD': None,
                    'MAD/MAE': None
                }
            mae = mean_absolute_error(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            mean_value = np.mean(y_true)
            baseline_pred = [mean_value] * len(y_true)
            mad = mean_absolute_error(y_true, baseline_pred)
            mad_mae_ratio = mad / mae if mae != 0 else None
            return {
                'total': total,
                'filtered': len(y_true),
                'MAE': mae,
                'RMSE': rmse,
                'MAD': mad,
                'MAD/MAE': mad_mae_ratio
            }
        else:
            y_true = []
            y_pred = []
            auc = None
            try:
                for t, p in zip(references, predictions):
                    if t in ['Null']:
                        continue
                    if t in ['Direct', 'True', True]:
                        y_true.append(1)
                    elif t in ['Indirect', 'False', False]:
                        y_true.append(0)
                    else:
                        continue

                    if p in ['Direct', 'True', True]:
                        y_pred.append(1)
                    elif p in ['Indirect', 'False', False]:
                        y_pred.append(0)
                    else:
                        y_true.pop()
                        continue
                auc = roc_auc_score(y_true, y_pred)
            except Exception:
                pass
            return {'AUC': auc}
