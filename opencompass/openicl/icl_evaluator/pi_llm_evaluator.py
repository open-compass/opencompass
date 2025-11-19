import json
import math
import re
from typing import Dict, List

from opencompass.registry import ICL_EVALUATORS

from .icl_base_evaluator import BaseEvaluator


@ICL_EVALUATORS.register_module()
class PILLMEvaluator(BaseEvaluator):
    """
    PI-LLM Evaluator with AUC (log base 1.5) scoring.

    Implements the exact scoring system from the HuggingFace dataset:
    https://huggingface.co/datasets/giantfish-fly/pi-llm

    Provides experiment-aware scoring:
    - Single-mode: exp_updates, exp_sequential → accuracy + auc_log1.5
    - Two-mode: exp_keys, exp_valuelength → accuracy + auc_log1.5
                                             + easy/hard breakdown

    Paper: https://arxiv.org/abs/2506.08184
    (ICML 2025 Workshop on Long-Context Foundation Models)
    """

    def __init__(self, log_base: float = 1.5) -> None:
        super().__init__()
        self.log_base = log_base

    def score(self,
              predictions: List,
              references: List,
              test_set: List[Dict] = None) -> dict:
        """
        Compute experiment-aware PI-LLM scores using AUC weighting.

        Returns different score structures based on experiment type:
        - Single-mode: {accuracy, auc_log1.5, total_samples}
        - Two-mode: {accuracy, auc_log1.5, auc_log1.5_easy,
                     auc_log1.5_hard, total_samples}
        """
        if len(predictions) != len(references):
            return {'error': 'predictions and references length mismatch'}

        if not test_set:
            return {'error': 'test_set required for metadata'}

        # Collect sample results with metadata
        results = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            accuracy = self.grade_pi_response(pred, ref)
            if accuracy is None:
                continue

            metadata = test_set[i] if i < len(test_set) else {}
            results.append({
                'accuracy': accuracy,
                'experiment': metadata.get('experiment', ''),
                'n_updates': metadata.get('n_updates', 2)
            })

        if not results:
            return {'error': 'no valid samples'}

        # Use the exact AUC function from HuggingFace
        return self.compute_pi_auc_score(results, self.log_base)

    def compute_pi_auc_score(self, results, log_base=1.5):
        """
        PI-LLM AUC score (PRIMARY: 'auc_log1.5').

        Uses log_base(n_updates) weights.
        - For two-mode experiments (keys/value length),
          also returns easy/hard AUCs.
        - For others (updates/sequential), returns a single overall AUC.

        This is the exact function from the HuggingFace dataset page.
        """
        if not results:
            return {'avg_accuracy': 0.0, 'auc_log1.5': 0.0, 'total_samples': 0}

        def wmean(samples):
            # weight = log_base(max(n_updates, 2)) to reflect difficulty
            ws = [
                math.log(max(s.get('n_updates', 2), 2), log_base)
                for s in samples
            ]
            denom = sum(ws)
            if denom:
                return sum(s['accuracy'] * w
                           for s, w in zip(samples, ws)) / denom
            else:
                return 0.0

        exp = results[0].get('experiment', '')
        avg = sum(s['accuracy'] for s in results) / len(results)
        overall = wmean(results)

        # Two-mode thresholds
        if 'exp_keys' in exp:
            easy_thr, hard_thr = 125, 350
        elif 'exp_valuelength' in exp:
            easy_thr, hard_thr = 4, 20
        else:
            # Single-mode path
            return {
                'avg_accuracy': avg,
                'auc_log1.5': overall,
                'total_samples': len(results)
            }

        easy = [s for s in results if s.get('n_updates', 0) <= easy_thr]
        hard = [s for s in results if s.get('n_updates', 0) >= hard_thr]

        return {
            'avg_accuracy': avg,
            'auc_log1.5': overall,  # PRIMARY metric
            'auc_log1.5_easy': wmean(easy) if easy else 0.0,
            'auc_log1.5_hard': wmean(hard) if hard else 0.0,
            'total_samples': len(results),
        }

    def extract_pieces_response_to_dict(self,
                                        model_output,
                                        probe_target='current'):
        """
        Extract the dictionary of key-value pairs from the model output.

        First extract using verbal language match, then using colon match.
        Merge the two dictionaries, prioritizing keys from the verbal match.
        """
        if len(model_output) == 0:
            return None
        if 'error code' in model_output.lower():
            return None
        if (model_output.startswith('error')
                or model_output.startswith('Error')):
            return None
        if (re.search(r'\berror\b', model_output, re.IGNORECASE)
                and (len(model_output) < 680)):
            return None

        # Remove backslashes and asterisks
        model_output = re.sub(r'\\(?!n)', '', model_output)
        model_output = re.sub(r'\*', '', model_output)

        dict_verbal_match = self._extract_verbal_matches(
            model_output, probe_target)
        dict_colon_match = self._extract_colon_matches(model_output)

        dict_merged = dict_colon_match.copy()
        dict_merged.update(dict_verbal_match)
        dict_merged.pop('key', None)

        return dict_merged

    def _extract_verbal_matches(self,
                                model_output: str,
                                probe_target='current'):
        """
        Extract key-value pairs using verbal patterns.

        Patterns like 'The current value of X is Y'
        """
        patterns = [
            r'(?:the)?\s*(?:most recent|final|last|latest|current|'
            r'up-to-date|asked|queried|specified)\s+(?:value|word|term)?'
            r'(?:s)?(?:\s+\w+){0,1}\s+(?:with|for|of|to)?\s+(?:the )?'
            r"(?:category|key)?\s*([\"'\[\<]?\w+(?:\s+\w+)?[\"'\]\>]?)\s+"
            r'(?:is|was)(?:\s*:\s*)?\s+'
            r"([\"'\[\<]?\w+(?:\s+\w+)?[\"'\]\>]?)(?=\n|[,.;:]|$)",
        ]

        dict_response = {}
        for pattern in patterns:
            matches = re.findall(pattern, model_output,
                                 re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match) >= 2:
                    key, value = match[0], match[1]
                    key = re.sub(r'[\*\'"""'
                                 r'\[\]\{\}\(\)\<\>]', '', key).strip()
                    value = re.sub(r'[\*\'"""'
                                   r'\[\]\{\}\(\)\<\>]', '', value).strip()
                    if key and value:
                        dict_response[key] = value
        return dict_response

    def _extract_colon_matches(self, model_output: str):
        """Extract key-value pairs using colon-separated patterns"""
        dict_response = {}
        lines = model_output.split('\n')
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = re.sub(r'[\*\'"""'
                                 r'\[\]\{\}\(\)\<\>]', '', parts[0]).strip()
                    value = re.sub(r'[\*\'"""'
                                   r'\[\]\{\}\(\)\<\>]', '', parts[1]).strip()
                    if key and value:
                        dict_response[key] = value
        return dict_response

    def grade_pi_response(self, response, answer_formatted):
        """
        Compute per-row accuracy for PI-LLM.

        Fraction of tracked keys answered with the last value.
        - Parses the ground truth JSON string (answer_formatted)
          into {key: last_value}.
        - Parses model output into {key: value} using robust extractors.
        - Returns (# of keys with exact value match) / (# of keys in GT).
        """
        try:
            # Parse ground truth JSON
            ground_truth = json.loads(answer_formatted)

            # Extract key-value pairs from model response
            response_dict = self.extract_pieces_response_to_dict(
                response, probe_target='current')

            if not isinstance(ground_truth, dict) or ground_truth is None:
                return 0.0
            if not isinstance(response_dict, dict) or response_dict is None:
                return 0.0

            keys = list(ground_truth.keys())
            if len(keys) == 0:
                return 0.0

            correct = sum(1 for k in keys
                          if response_dict.get(k) == ground_truth.get(k))
            return correct / len(keys)
        except Exception:
            return 0.0
