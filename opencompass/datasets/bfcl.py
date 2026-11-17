"""BFCL (Berkeley Function Calling Leaderboard) simple subset.

Data: place the two files shipped by the benchmark under ``data/bfcl/``::

    BFCL_v3_simple.json           # questions, JSONL
    BFCL_v3_simple_answer.json    # acceptable calls, JSONL

They are published at
https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
(``BFCL_v3_simple.json`` and ``possible_answer/BFCL_v3_simple.json``).

The prompt asks the model for one JSON call, which is the prompt-based mode of
BFCL; the evaluator checks the function name and every reference argument.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

PROMPT_HEADER = ('You have access to the following functions:\n'
                 '{functions}\n\n'
                 'Given the user request, call exactly one function.\n'
                 'Reply with ONLY a JSON object of the form\n'
                 '{{"name": "<function name>", "arguments": {{...}}}}\n'
                 'and nothing else.\n')

PROMPT_FOOTER = '\nUser: {question}\n'


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    """BFCL ships one JSON object per line, not a JSON array."""
    with open(path, encoding='utf-8') as fin:
        return [json.loads(line) for line in fin if line.strip()]


def _render_functions(functions: List[Dict[str, Any]]) -> str:
    """Render the function schemas the way BFCL's own prompt mode does."""
    return json.dumps(functions, ensure_ascii=False, indent=2)


def _first_user_message(question: List[List[Dict[str, str]]]) -> str:
    """BFCL stores a list of turns per sample; the simple subset has one."""
    for turn in question:
        for message in turn:
            if message.get('role') == 'user':
                return message.get('content', '')
    return ''


@LOAD_DATASET.register_module()
class BFCLDataset(BaseDataset):
    """BFCL (Berkeley Function Calling Leaderboard) simple subset.

    Each sample ships the user request plus the JSON schemas of the available
    functions. The prompt asks the model to answer with a single JSON call,
    which is BFCL's own prompt-based mode; the evaluator then compares that
    call against the reference call(s) shipped with the benchmark.

    Args:
        path (str): path to ``BFCL_v3_simple.json`` (JSONL).
        answer_path (str): path to ``possible_answer/BFCL_v3_simple.json``.
    """

    @staticmethod
    def load(path: str, answer_path: str, **kwargs) -> Dataset:
        path = get_data_path(path, local_mode=True)
        answer_path = get_data_path(answer_path, local_mode=True)

        questions = _load_jsonl(path)
        answers = {item['id']: item for item in _load_jsonl(answer_path)}

        raw_data = []
        for item in questions:
            sample_id = item['id']
            answer = answers.get(sample_id)
            if answer is None:
                # A question without a reference can never be scored, so it is
                # skipped instead of silently counting as wrong.
                continue

            question = _first_user_message(item['question'])
            prompt = PROMPT_HEADER.format(
                functions=_render_functions(item['function'])) + \
                PROMPT_FOOTER.format(question=question)

            raw_data.append({
                'id':
                sample_id,
                'prompt':
                prompt,
                'gts':
                json.dumps(answer['ground_truth'], ensure_ascii=False),
            })

        return Dataset.from_list(raw_data)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    fence = re.match(r'^```(?:json)?\s*(.*?)\s*```$', text, re.S)
    return fence.group(1) if fence else text


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Return the first JSON object in ``text``, or None.

    Models sometimes wrap the call in prose or a code fence, so the first
    balanced ``{...}`` block is tried instead of requiring the whole reply to
    be JSON.
    """
    text = _strip_code_fence(text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find('{')
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = text.find('{', start + 1)
    return None


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _value_match(predicted: Any, allowed: Any) -> bool:
    """Whether ``predicted`` equals one of the acceptable values.

    ``allowed`` is the list of acceptable values BFCL ships for an argument.
    Numbers are compared numerically (``5`` == ``5.0`` == ``"5"``), strings
    after stripping, and containers recursively.
    """
    if not isinstance(allowed, list):
        allowed = [allowed]

    for candidate in allowed:
        if predicted == candidate:
            return True

        predicted_number = _as_number(predicted)
        candidate_number = _as_number(candidate)
        if predicted_number is not None and candidate_number is not None:
            if predicted_number == candidate_number:
                return True
            continue

        if isinstance(predicted, str) and isinstance(candidate, str):
            if predicted.strip() == candidate.strip():
                return True

        if isinstance(predicted, list) and isinstance(candidate, list):
            if len(predicted) == len(candidate) and all(
                    _value_match(p, c) for p, c in zip(predicted, candidate)):
                return True

        if isinstance(predicted, dict) and isinstance(candidate, dict):
            if predicted.keys() == candidate.keys() and all(
                    _value_match(predicted[key], candidate[key])
                    for key in predicted):
                return True

    return False


def _call_matches(predicted: Dict[str, Any], reference: Dict[str,
                                                             Any]) -> bool:
    """Whether one predicted call matches one acceptable reference call.

    Every argument listed in the reference must be present and acceptable.
    Extra arguments the model adds are ignored, which matches the leniency of
    the official checker for optional parameters.
    """
    if not isinstance(predicted, dict) or len(reference) != 1:
        return False

    (function_name, expected_arguments), = reference.items()
    predicted_name = predicted.get('name') or predicted.get('function')
    if predicted_name != function_name:
        return False

    arguments = predicted.get('arguments', predicted.get('parameters', {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return False
    if not isinstance(arguments, dict):
        return False

    for argument, allowed in expected_arguments.items():
        if argument not in arguments:
            # BFCL marks an optional argument that the caller may omit with
            # an empty string among its acceptable values.
            if _value_match('', allowed):
                continue
            return False
        if not _value_match(arguments[argument], allowed):
            return False
    return True


@ICL_EVALUATORS.register_module()
class BFCLEvaluator(BaseEvaluator):
    """Exact-match accuracy for the BFCL simple subset.

    A prediction counts as correct when it names the reference function and
    every reference argument matches one of the acceptable values.

    Notes:
        This is a strict subset of the official BFCL scorer, which also runs
    an AST checker and per-argument type validation. Long context, parallel,
    multiple and irrelevance subsets are not covered here.
    """

    def score(self, predictions: List[str],
              references: List[str]) -> Dict[str, Any]:
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        details = []
        correct = 0
        parse_failures = 0

        for index, (prediction,
                    reference) in enumerate(zip(predictions, references)):
            try:
                acceptables = json.loads(reference)
            except (TypeError, json.JSONDecodeError):
                acceptables = []

            call = _extract_json_object(prediction or '')
            if call is None:
                parse_failures += 1
                matched = False
            else:
                matched = any(
                    _call_matches(call, acceptable)
                    for acceptable in acceptables
                    if isinstance(acceptable, dict))
            correct += int(matched)
            details.append({
                'index': index,
                'correct': matched,
            })

        total = len(predictions)
        return {
            'accuracy': 100 * correct / total if total else 0.0,
            'parse_failure_rate':
            100 * parse_failures / total if total else 0.0,
            'details': details,
        }
