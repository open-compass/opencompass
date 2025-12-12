# NC-I2S NC-S2I task
# https://github.com/OSU-NLP-Group/LLM4Chem

import json
import re

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from .config import TASK_TAGS, TASKS_WITH_SEMICOLON_REPLACE
from .utils.metrics import (calculate_boolean_metrics,
                            calculate_formula_metrics,
                            calculate_number_metrics, calculate_smiles_metrics,
                            calculate_text_metrics)

from opencompass.utils import get_data_path
import os

@LOAD_DATASET.register_module()
class LLM4ChemDataset(BaseDataset):

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
            test_data = random.sample(test_data, 50)
            random.seed()

        dataset = DatasetDict({
            'train': Dataset.from_list(train_data),
            'test': Dataset.from_list(test_data)
        })
        return dataset


def extract_answer_part(outputs, left_tag, right_tag, mode='tag'):
    assert mode in ('tag', 'direct')

    assert isinstance(outputs, list)
    answers = []
    for text in outputs:
        if mode == 'direct' or (left_tag is None and right_tag is None):
            text = text.replace('<unk>', '').replace('</s>', '').strip()
            answers.append(text.strip())
            continue

        left_tag_pos = text.find(left_tag)
        if left_tag_pos == -1:
            answers.append('')
            continue
        right_tag_pos = text.find(right_tag)
        if right_tag_pos == -1:
            answers.append('')
            continue
        text = text[left_tag_pos + len(left_tag):right_tag_pos].strip()
        answers.append(text)
    return answers


@TEXT_POSTPROCESSORS.register_module('LLM4Chem_postprocess')
def LLM4Chem_postprocess(text, task, *args, **kwargs):
    # 删除 <think> </think> 里的内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    replace_semicolon = task in TASKS_WITH_SEMICOLON_REPLACE
    pred = extract_answer_part([text], *(TASK_TAGS[task]), mode='tag')[0]
    # task in TASKS_WITH_SEMICOLON_REPLACE needs semicolon
    # replaced with a period
    if replace_semicolon:
        pred = pred.replace(';', '.')
    # no matched tag
    if pred == '':
        tag = TASK_TAGS[task][0]

        if (tag == '<BOOLEAN>'):
            # 找到 text 的最后一个 yes/true/no/false，不区分大小写
            ans = re.findall(r'\b(?:yes|true|no|false)\b', text, re.IGNORECASE)
            if ans:
                # if ans[-1] 是 yes/true
                if ans[-1].lower() in ('yes', 'true'):
                    return 'Yes'
                else:
                    return 'No'
            else:
                return ''

        if (tag == '<NUMBER>'):
            # 找到 text 的最后一个数字
            # 去掉 text 里 <SMILES> </SMILES> 里的内容
            text_2 = re.sub(r'<SMILES>.*?</SMILES>', '', text, flags=re.DOTALL)
            ans = re.findall(r'-?\d*\.\d+|-?\d+', text_2)
            if ans:
                return ans[-1]
            else:
                return ''

        if (tag == '<MOLFORMULA>'):
            # 找到 text 的最后一个化学式
            ans = re.findall(
                r'[\[\(]?[A-Z][a-z]?\d*(?:\([A-Za-z0-9]+\)\d*)?[\]\)]?'
                r'(?:[A-Z][a-z]?\d*|\([^\)]+\)\d*|\[[^\]]+\]\d*)'
                r'*(?:[+-]{1,2})?(?:·\d*[A-Z][a-z]?\d*)*', text)
            if ans:
                return ans[-1]
            else:
                return ''

    # print(f"prediction: {pred}")
    return pred


class LLM4Chem_Evaluator(BaseEvaluator):

    def __init__(self, task, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        if not isinstance(predictions[0], list):
            predictions = [[pred] for pred in predictions]
        if not isinstance(references[0], list):
            references = [[ref] for ref in references]

        task = self.task
        pred_list = predictions
        gold_list = references

        if task in ('property_prediction-esol', 'property_prediction-lipo',
                    'property_prediction-bbbp', 'property_prediction-clintox',
                    'property_prediction-hiv', 'property_prediction-sider'):
            # set pred_list to [length * 1]
            pred_list = [[pred[0]] for pred in pred_list]

        if task in ('forward_synthesis', 'molecule_generation',
                    'name_conversion-i2s'):
            r = calculate_smiles_metrics(pred_list, gold_list)
        elif task in ('retrosynthesis', ):
            r = calculate_smiles_metrics(pred_list,
                                         gold_list,
                                         metrics=('exact_match', 'fingerprint',
                                                  'multiple_match'))
        elif task in ('molecule_captioning', ):
            r = calculate_text_metrics(
                pred_list,
                gold_list,
                text_model='allenai/scibert_scivocab_uncased',
                text_trunc_length=2048,
            )
        elif task in ('name_conversion-i2f', 'name_conversion-s2f'):
            r = calculate_formula_metrics(pred_list,
                                          gold_list,
                                          metrics=('element_match', ))
        elif task in ('name_conversion-s2i', ):
            r = calculate_formula_metrics(pred_list,
                                          gold_list,
                                          metrics=('split_match', ))
        elif task in ('property_prediction-esol', 'property_prediction-lipo'):
            r = calculate_number_metrics(pred_list, gold_list)
        elif task in ('property_prediction-bbbp',
                      'property_prediction-clintox', 'property_prediction-hiv',
                      'property_prediction-sider'):
            r = calculate_boolean_metrics(pred_list, gold_list)
        else:
            raise ValueError(task)

        if 'num_t1_exact_match' in r and 'num_all' in r:
            # 100%, 2 位小数
            r['top1_exact_match'] = round(
                r['num_t1_exact_match'] / r['num_all'] * 100, 2)
        if 'num_t5_exact_match' in r and 'num_all' in r:
            # 100%, 2 位小数
            r['top5_exact_match'] = round(
                r['num_t5_exact_match'] / r['num_all'] * 100, 2)
        if 'num_t1_ele_match' in r and 'num_all' in r:
            # 100%, 2 位小数
            r['top1_ele_match'] = round(
                r['num_t1_ele_match'] / r['num_all'] * 100, 2)
        if 'num_correct' in r and 'num_all' in r:
            r['accuracy'] = round(r['num_correct'] / r['num_all'] * 100, 2)
        if 'num_t1_split_match' in r and 'num_all' in r:
            # 100%, 2 位小数
            r['top1_split_match'] = round(
                r['num_t1_split_match'] / r['num_all'] * 100, 2)
        if 'num_t5_split_match' in r and 'num_all' in r:
            # 100%, 2 位小数
            r['top5_split_match'] = round(
                r['num_t5_split_match'] / r['num_all'] * 100, 2)

        return r
