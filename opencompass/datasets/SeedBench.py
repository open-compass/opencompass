import random
import re
from os import environ
from typing import List

import datasets
import jieba
import numpy as np
from rouge_chinese import Rouge

from opencompass.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from opencompass.registry import (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class SeedBenchDataset(BaseDataset):

    @staticmethod
    def load(data_files: str,
             path: str,
             split: str = None,
             **kwargs) -> datasets.Dataset:

        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(path,
                                     subset_name='default',
                                     split=split,
                                     data_files=data_files,
                                     **kwargs)
        else:
            dataset = datasets.load_dataset(path,
                                            data_files=data_files,
                                            **kwargs)

        if split is None:
            split = list(dataset.keys())[0]

        if split not in dataset:
            raise ValueError(f"Split '{split}' not found. \
                    Available splits: {list(dataset.keys())}")

        return dataset[split]


class F1Evaluator(BaseEvaluator):
    """F1 Score evaluator for multiple choice questions.

    Args:
        seed (int): Seed for randomness, ensuring reproducibility.
        Defaults to 0.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        super().__init__()

    def _preprocess(self, predictions: List, references: List) -> dict:
        return {
            'predictions': predictions,
            'references': references,
        }

    def _postprocess(self, scores: dict) -> dict:
        return scores

    def score(self, predictions: List, references: List) -> dict:
        random_state = random.getstate()
        np_random_state = np.random.get_state()
        details = []

        random.seed(self.seed)
        np.random.seed(self.seed)

        if len(predictions) != len(references):
            return {
                'error':
                'predictions and references have different '
                f'length. len(predictions): {len(predictions)}, '
                f'len(references): {len(references)}'
            }

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for hyp, ref in zip(predictions, references):
            hyp = re.sub(r'[^A-Da-d,]+', '', hyp.lower())
            ref = re.sub(r'[^A-Da-d,]+', '', ref.lower())
            ref_set = set(ref.split(','))
            hyp_set = set(hyp.split(','))
            ref_set = {r.strip() for r in ref_set}
            hyp_set = {h.strip() for h in hyp_set}

            sample_tp = len(hyp_set.intersection(ref_set))
            sample_fp = len(hyp_set - ref_set)
            sample_fn = len(ref_set - hyp_set)
            true_positives += sample_tp
            false_positives += sample_fp
            false_negatives += sample_fn
            sample_precision = sample_tp / (sample_tp + sample_fp) if (
                sample_tp + sample_fp) > 0 else 0
            sample_recall = sample_tp / (sample_tp + sample_fn) if (
                sample_tp + sample_fn) > 0 else 0
            sample_f1 = (2 * sample_precision * sample_recall) / (
                sample_precision + sample_recall) if (sample_precision +
                                                      sample_recall) > 0 else 0
            details.append({
                'pred': hyp,
                'answer': ref,
                'correct': sample_f1 * 100
            })

        precision = true_positives / (true_positives + false_positives) if (
            true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (
            true_positives + false_negatives) > 0 else 0
        f1 = (2 * precision *
              recall) / (precision + recall) if (precision + recall) > 0 else 0

        result = {
            'F1Score': f1 * 100,  # 总体 F1 分数
            'details': details
        }
        random.setstate(random_state)
        np.random.set_state(np_random_state)
        return self._postprocess(result)


@ICL_EVALUATORS.register_module()
class F1ScoreEvaluator(F1Evaluator):
    """F1 Score evaluator for multiple choice questions."""

    def __init__(self) -> None:
        super().__init__()


# 定义自己的多选后处理逻辑（输入回答为：ABC ---> A,B,C)
@TEXT_POSTPROCESSORS.register_module('my_multiple_select_postprocess')
def my_multiple_select_postprocess(text: str) -> str:
    selected_options = [t for t in text if t.isupper()]
    selected_options = sorted(set(selected_options))
    res = ', '.join(selected_options)
    return res


class AverageRougeEvaluator(BaseEvaluator):
    """Average Rouge Score evaluator for fill-in-the-blank tasks.

    Args:
        seed (int): Seed for randomness, ensuring reproducibility.
        Defaults to 0.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        super().__init__()

    def _preprocess(self, predictions: List, references: List) -> dict:
        pattern = r'(正确答案[:：]|correct answer[:：])'
        cleaned_predictions = [
            re.sub(pattern, '', pred, flags=re.IGNORECASE).strip()
            for pred in predictions
        ]

        return {
            'predictions': cleaned_predictions,
            'references': references,
        }

    def _postprocess(self, scores: dict) -> dict:
        return scores

    def score(self, predictions: List, references: List) -> dict:

        def rouge_score(hyps, refs):
            assert (len(hyps) == len(refs))
            hyps = [' '.join(jieba.cut(h)) for h in hyps]
            hyps = [h if h.strip() != '' else '无内容' for h in hyps]
            refs = [' '.join(jieba.cut(r)) for r in refs]
            rouge_scores = Rouge().get_scores(hyps, refs)
            rouge_ls = [score['rouge-l']['f'] for score in rouge_scores]
            average_rouge_l = sum(rouge_ls) / len(rouge_ls)
            return {'score': average_rouge_l * 100}

        random_state = random.getstate()
        np_random_state = np.random.get_state()
        details = []
        random.seed(self.seed)
        np.random.seed(self.seed)

        if len(predictions) != len(references):
            return {
                'error':
                'predictions and references have different '
                f'length. len(predictions): {len(predictions)}, '
                f'len(references): {len(references)}'
            }

        preprocessed_data = self._preprocess(predictions, references)
        hyps, refs = preprocessed_data['predictions'], preprocessed_data[
            'references']

        scores = []
        for i in range(len(hyps)):
            refs[i] = refs[i].replace('，', ',')
            word_level_refs = refs[i].split(',')
            word_level_refs = [r.strip() for r in word_level_refs]
            if len(word_level_refs) == 1:
                word_level_hyps = [hyps[i]]
            else:
                word_level_hyps = hyps[i].split(',')
                word_level_hyps = [h.strip() for h in word_level_hyps]

                if len(word_level_hyps) < len(word_level_refs):
                    word_level_hyps += ['无内容'] * (len(word_level_refs) -
                                                  len(word_level_hyps))
                else:
                    word_level_hyps = word_level_hyps[:len(word_level_refs)]

            sample_score = rouge_score(word_level_hyps,
                                       word_level_refs)['score']
            scores.append(sample_score)
            details.append({
                'pred': word_level_hyps,
                'answer': word_level_refs,
                'correct': sample_score
            })

        average_score = sum(scores) / len(scores)
        result = {'AvgRougeScore': average_score, 'details': details}
        random.setstate(random_state)
        np.random.set_state(np_random_state)

        return self._postprocess(result)


@ICL_EVALUATORS.register_module()
class AverageRougeScoreEvaluator(AverageRougeEvaluator):
    """Average Rouge Score evaluator."""

    def __init__(self) -> None:
        super().__init__()


class AccScoreStrEvaluator(BaseEvaluator):
    """Accuracy evaluator based on string matching.

    Args:
        seed (int): Seed for randomness, ensuring reproducibility.
        Defaults to 0.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        super().__init__()

    def _preprocess(self, predictions: List, references: List) -> dict:
        return {
            'predictions': predictions,
            'references': references,
        }

    def _postprocess(self, scores: dict) -> dict:
        return scores

    def score(self, predictions: List, references: List) -> dict:
        random_state = random.getstate()
        np_random_state = np.random.get_state()
        details = []
        random.seed(self.seed)
        np.random.seed(self.seed)

        if len(predictions) != len(references):
            return {
                'error':
                'predictions and references have different '
                f'length. len(predictions): {len(predictions)}, '
                f'len(references): {len(references)}'
            }

        preprocessed_data = self._preprocess(predictions, references)

        correct = 0
        for hyp, ref in zip(preprocessed_data['predictions'],
                            preprocessed_data['references']):
            is_correct = 1 if ref.strip().lower() in hyp.strip().lower() else 0
            correct += is_correct
            details.append({'pred': hyp, 'answer': ref, 'correct': is_correct})

        accuracy = correct / len(predictions)
        result = {'ACCStrScore': accuracy * 100, 'details': details}
        random.setstate(random_state)
        np.random.set_state(np_random_state)

        return self._postprocess(result)


@ICL_EVALUATORS.register_module()
class AccScoreStr_Evaluator(AccScoreStrEvaluator):
    """Accuracy evaluator wrapper for the AccScoreEvaluator."""

    def __init__(self) -> None:
        super().__init__()
