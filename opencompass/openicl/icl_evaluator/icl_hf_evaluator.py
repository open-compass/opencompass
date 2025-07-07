import os
import random
from typing import List

import evaluate
import numpy as np
from datasets import Dataset

from opencompass.registry import ICL_EVALUATORS

from .icl_base_evaluator import BaseEvaluator


class HuggingfaceEvaluator(BaseEvaluator):
    """Use huggingface evaluate module to calculate the target metrics.

    Args:
        metric (str): Metric name in evaluate module.
        seed (int): There exists some randomness during the calculation of some
            metrics, thus we set a fixed random seed for reproducing. Defaults
            to 0.
    """

    def __init__(self, metric: str, seed: int = 0) -> None:
        self.metric = metric
        self.seed = seed
        super().__init__()

    def _preprocess(self, predictions: List, references: List) -> dict:
        """Preprocess the final predictions and references to needed format.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: preprocessed results.
        """
        return {
            'predictions': predictions,
            'references': references,
        }

    def _postprocess(self, scores: dict) -> dict:
        """Postprocess for final scores.

        Args:
            scores (dict): Dict of calculated scores of metrics.

        Returns:
            dict: postprocessed scores.
        """
        return scores

    def score(self, predictions: List, references: List) -> dict:
        """Calculate scores.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: calculated scores.
        """
        random_state = random.getstate()
        np_random_state = np.random.get_state()

        random.seed(self.seed)
        np.random.seed(self.seed)
        if len(predictions) != len(references):
            return {
                'error':
                'predictions and references have different '
                f'length. len(predictions): {len(predictions)}, '
                f'len(references): {len(references)}'
            }
        # use codes pre-downloaded to opencompass repo, avoid downloading
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'hf_metrics', self.metric + '.py')
        if os.path.exists(local_path):
            metric = evaluate.load(local_path)
        else:
            metric = evaluate.load(self.metric)
        scores = metric.compute(**self._preprocess(predictions, references))
        result = self._postprocess(scores)
        random.setstate(random_state)
        np.random.set_state(np_random_state)
        return result


@ICL_EVALUATORS.register_module()
class AccEvaluator(HuggingfaceEvaluator):
    """Accuracy evaluator."""

    def __init__(self) -> None:
        super().__init__(metric='accuracy')

    def _preprocess(self, predictions: List, references: List) -> dict:
        """Preprocess the final predictions and references to needed format.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: preprocessed results.
        """
        mapping_to_int_dict = {
            label: idx
            for idx, label in enumerate(set(map(str, references)))
        }
        pred_set = set(predictions)
        for pred in pred_set:
            if str(pred) not in mapping_to_int_dict.keys():
                mapping_to_int_dict[str(pred)] = len(mapping_to_int_dict)
        golds = [mapping_to_int_dict[str(gold)] for gold in references]
        preds = [mapping_to_int_dict[str(pred)] for pred in predictions]
        return {
            'predictions': preds,
            'references': golds,
        }

    def _postprocess(self, scores: dict) -> dict:
        """Postprocess for final scores.

        Args:
            scores (dict): Dict of calculated scores of metrics.

        Returns:
            dict: postprocessed scores.
        """
        scores['accuracy'] *= 100
        return scores


@ICL_EVALUATORS.register_module()
class AccContaminationEvaluator(AccEvaluator):
    """Accuracy evaluator."""

    def score(self, predictions: List, references: List,
              test_set: Dataset) -> dict:
        # group the predictions and references by their contamination status
        clean_predictions, clean_references = [], []
        input_contaminated_predictions, input_contaminated_references = [], []
        input_and_label_contaminated_predictions, \
            input_and_label_contaminated_references = [], []
        for pred, ref, is_clean in zip(predictions, references,
                                       test_set['is_clean']):
            if is_clean == 'clean':
                clean_predictions.append(pred)
                clean_references.append(ref)
            elif is_clean == 'input contamination':
                input_contaminated_predictions.append(pred)
                input_contaminated_references.append(ref)
            elif is_clean == 'input-and-label contamination':
                input_and_label_contaminated_predictions.append(pred)
                input_and_label_contaminated_references.append(ref)
        clean_results = super().score(clean_predictions, clean_references)
        input_contaminated_results = super().score(
            input_contaminated_predictions, input_contaminated_references)
        input_and_label_contaminated_results = super().score(
            input_and_label_contaminated_predictions,
            input_and_label_contaminated_references)

        # rename the keys of the results, add 'clean, 'input contaminated',
        # 'input-and-label contaminated' as prefixes
        clean_results = {f'{k} - clean': v for k, v in clean_results.items()}
        input_contaminated_results = {
            f'{k} - input contaminated': v
            for k, v in input_contaminated_results.items()
        }
        input_and_label_contaminated_results = {
            f'{k} - input-and-label contaminated': v
            for k, v in input_and_label_contaminated_results.items()
        }
        return {
            **clean_results,
            **input_contaminated_results,
            **input_and_label_contaminated_results
        }


@ICL_EVALUATORS.register_module()
class RougeEvaluator(HuggingfaceEvaluator):
    """Rouge evaluator.

    Note: this evaluator is not suitable for chinese datasets.
    """

    def __init__(self) -> None:
        super().__init__(metric='rouge')

    def _postprocess(self, scores: dict) -> dict:
        """Postprocess for final scores.

        Args:
            scores (dict): Dict of calculated scores of metrics.

        Returns:
            dict: postprocessed scores.
        """
        return {k: v * 100 for k, v in scores.items()}


@ICL_EVALUATORS.register_module()
class BleuEvaluator(HuggingfaceEvaluator):
    """Bleu evaluator."""

    def __init__(self) -> None:
        super().__init__(metric='sacrebleu')


class BleuFloresEvaluator(HuggingfaceEvaluator):
    """Bleu evaluator using flores200 tokenize."""

    def __init__(self) -> None:
        super().__init__(metric='sacrebleu')

    def _preprocess(self, predictions: List, references: List) -> dict:
        return {
            'predictions': predictions,
            'references': references,
            'tokenize': 'flores200',
        }


@ICL_EVALUATORS.register_module()
class MccEvaluator(AccEvaluator):
    """Matthews correlation evaluator."""

    def __init__(self) -> None:
        super(AccEvaluator, self).__init__(metric='matthews_correlation')

    def _postprocess(self, scores: dict) -> dict:
        """Postprocess for final scores.

        Args:
            scores (dict): Dict of calculated scores of metrics.

        Returns:
            dict: postprocessed scores.
        """
        scores['matthews_correlation'] *= 100
        return scores


@ICL_EVALUATORS.register_module()
class SquadEvaluator(HuggingfaceEvaluator):
    """Squad evaluator."""

    def __init__(self) -> None:
        super().__init__(metric='squad')

    def _preprocess(self, predictions: List, references: List) -> dict:
        """Preprocess the final predictions and references to needed format.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: preprocessed results.
        """
        p_list = [{
            'prediction_text': pred.split('\n')[0],
            'id': str(i)
        } for i, pred in enumerate(predictions)]
        r_list = [{
            'answers': {
                'answer_start': [0],
                'text': [ref]
            },
            'id': str(i)
        } for i, ref in enumerate(references)]
        return {
            'predictions': p_list,
            'references': r_list,
        }

    def _postprocess(self, scores: dict) -> dict:
        """Postprocess for final scores.

        Args:
            scores (dict): Dict of calculated scores of metrics.

        Returns:
            dict: postprocessed scores.
        """
        return scores['f1']


@ICL_EVALUATORS.register_module()
class EDAccEvaluator(AccEvaluator):
    """Edit distance based accuracy evaluator.

    This implementation requires the un-postprocessed outputs from the model,
    and the reference list where each item is structured as:

    .. code-block:: python

        {
            'candidates': [],  # a list of informative answer candidates
            'label': 0,  # the index of the gold answer
        }

    It always matches the model's output to a valid answer with the citerion
    as the minimum editing distance.
    """

    def __init__(self) -> None:
        super().__init__()
        from rapidfuzz.distance import Levenshtein
        self.dist = Levenshtein.distance

    def _preprocess(self, predictions: List, references: List) -> dict:
        """Preprocess the final predictions and references to needed format.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: preprocessed results.
        """

        preds = []
        golds = []

        for i in range(len(predictions)):
            pred, ref = predictions[i], references[i]
            dists = []
            for cands in ref['candidates']:
                if isinstance(cands, str):
                    d = self.dist(pred, cands)
                else:
                    d = np.min([self.dist(pred, cand) for cand in cands])
                dists.append(d)
            preds.append(np.argmin(dists))
            golds.append(ref['label'])

        return {
            'predictions': preds,
            'references': golds,
        }


@ICL_EVALUATORS.register_module()
class AccwithDetailsEvaluator(BaseEvaluator):

    def score(self, predictions, references, origin_prompt) -> dict:

        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length.'}

        details = {}
        correct, total = 0, 0
        for index, (pred, ref) in enumerate(zip(predictions, references)):
            is_correct = pred == ref
            correct += is_correct
            details[str(index)] = {
                'prompt': origin_prompt[index],
                'pred': pred,
                'refr': ref,
                'is_correct': is_correct,
            }
            total += 1

        results = {'accuracy': correct / total * 100, 'details': details}

        return results
