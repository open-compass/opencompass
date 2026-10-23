# flake8: noqa

import json
import os
import re
from typing import Union

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from sklearn.metrics import matthews_corrcoef

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path


@LOAD_DATASET.register_module()
class GUE_Dataset(BaseDataset):

    @staticmethod
    def load(path, task, mini_set=False):

        # if (hf_hub is True):
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

        def augment_output(data):
            for item in data:
                label = item.get('meta_data', {}).get('label', '')
                item['output'] += f' The prediction result is {label}.'
            return data

        train_data = augment_output(train_data[:5])
        test_data = augment_output(test_data)
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


def remove_think_tags(text: str) -> str:
    if '<think>' not in text:
        return text
    if '</think>' not in text:
        return ''
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)


@TEXT_POSTPROCESSORS.register_module()
def GUE_postprocessor(text: Union[str, None]) -> str:
    if not isinstance(text, str):
        return ''

    text = text.strip()
    text = remove_think_tags(text)

    if text == '':
        return ''

    match = re.search(r'\bThe prediction result is\s+(positive|negative)\b',
                      text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    positive_patterns = [
        r'\bpositive\b',
        r'\bpositively\b',
        r'\bpresence\b',
        r'\bdetected\b',
        r'\bidentified\b',
        r'\bidentifiable\b',
        r'\bfound\b',
        r'\byes\b',
        r'\blocated\b',
        r'\bdetectable\b',
        r'\bobservable\b',
        r'\bevident\b',
        r'\babsolutely\b',
        r'\baffirmative\b',
        r'\bcan\b',
        r'\baffirm\b',
        r'\bconfirm\b',
        r'\bconfirms\b',
        r'\breveals\b',
        r'\bexistence\b',
        r'\bcertainly\b',
        r'\bconsistent\b',
        r'\brecognizable\b',
        r'\bshows core\b',
        r'\bshows promoter\b',
        r'\bshows characteristic\b',
        r'\bevidenced by\b',
        r'\bseeing characteristic patterns\b',
        r'\bincludes\b',
        r'\bcontains sequences\b',
        r'\bexhibits clear\b',
        r'\bcontains transcription\b',
        r'\bexhibits sequences\b',
        r'\bclearly contains\b',
        r'\brecognized\b',
        r'\bexhibits features\b',
        r'\bcontains regulatory\b',
        r'\bshows clear\b',
        r'\bdisplays\b',
        r'\bdefinitely has\b',
        r'\bexhibits patterns\b',
        r'\bclear evidence\b',
        r'\bcontains a\b',
        r'\byep\b',
        r'\bcontains sites\b',
        r'\bshows sequences\b',
    ]

    negative_patterns = [
        r'\bnegative\b',
        r'\bno\b',
        r'\babsence\b',
        r'\bnot\b',
        r'\bcannot\b',
        r'\bfails\b',
        r'\babsent\b',
        r'\blacks\b',
    ]

    for pattern in negative_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return 'negative'

    for pattern in positive_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return 'positive'

    return ''


class GUE_Evaluator(BaseEvaluator):

    def score(self, predictions, references):

        def normalize(label):
            label = label.strip().lower()
            if label == 'positive':
                return 1
            elif label == 'negative':
                return 0
            else:
                return None

        total_count = len(predictions)

        if isinstance(predictions[0], list):
            predictions = [p[0] for p in predictions]

        pred_bin_all = [
            1 if p.strip().lower() == 'positive' else 0 for p in predictions
        ]
        ref_bin_all = [
            1 if r.strip().lower() == 'positive' else 0 for r in references
        ]
        mcc_all = matthews_corrcoef(ref_bin_all, pred_bin_all)

        filtered_pred = []
        filtered_ref = []
        skipped = 0

        for p, r in zip(predictions, references):
            p_norm = normalize(p)
            r_norm = normalize(r)
            if p_norm is None or r_norm is None:
                skipped += 1
                continue
            filtered_pred.append(p_norm)
            filtered_ref.append(r_norm)

        if filtered_pred:
            mcc_filtered = matthews_corrcoef(filtered_ref, filtered_pred)
        else:
            mcc_filtered = 0.0

        return {
            'matthews_correlation_all': mcc_all * 100,
            'matthews_correlation_filtered': mcc_filtered * 100,
            'non_pos_neg_count': skipped,
            'total_count': total_count
        }
