"""
DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs
Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, Matt Gardner
https://arxiv.org/abs/1903.00161
"""

import gzip
import json
import random
import re
import string
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import common
from .common import ANSWER_PATTERN, HTML_JINJA
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult
"""
From here through _normalize_answer was originally copied from:
https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
Then cleaned up and modified a bit.

The rest was originally copied from https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc
/eval/drop_eval.py
"""


def _remove_articles(text: str) -> str:
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)


def _white_space_fix(text: str) -> str:
    return ' '.join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return ''.join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _lower(text: str) -> str:
    return text.lower()


def _tokenize(text: str) -> List[str]:
    return re.split(' |-', text)


def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    parts = [
        _white_space_fix(
            _remove_articles(_normalize_number(_remove_punc(_lower(token)))))
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = ' '.join(parts).strip()
    return normalized


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    else:
        return text


def _answer_to_bags(
    answer: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]],
                gold: List[Set[str]]) -> List[float]:
    """Takes gold and predicted answer sets and first finds the optimal 1-1
    alignment between them and gets maximum metric values over all the
    answers."""
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index,
                       pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = ((2 * precision * recall) / (precision + recall)
          if not (precision == 0.0 and recall == 0.0) else 0.0) * 100
    return f1


def _match_numbers_if_present(gold_bag: Set[str],
                              predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def get_drop_metrics(
        predicted: Union[str, List[str], Tuple[str, ...]],
        gold: Union[str, List[str], Tuple[str, ...]]) -> Tuple[float, float]:
    """Takes a predicted answer and a gold answer (that are both either a
    string or a list of strings), and returns exact match and the DROP F1
    metric for the prediction.

    If you are
    writing a script for evaluating objects in memory (say, the output of predictions during
    validation, or while training), this is the function you want to call, after using
    :func:`answer_json_to_strings` when reading the gold answer from the released data file.
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(
            predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1


def answer_json_to_strings(
        answer: Dict[str, Any]) -> Tuple[Tuple[str, ...], str]:
    """Takes an answer JSON blob from the DROP data release and converts it
    into strings used for evaluation."""
    if 'number' in answer and answer['number']:
        return tuple([str(answer['number'])]), 'number'
    elif 'spans' in answer and answer['spans']:
        return tuple(
            answer['spans']), 'span' if len(answer['spans']) == 1 else 'spans'
    elif 'date' in answer:
        return (
            tuple([
                '{0} {1} {2}'.format(answer['date']['day'],
                                     answer['date']['month'],
                                     answer['date']['year']).strip()
            ]),
            'date',
        )
    else:
        raise ValueError(
            f'Answer type not found, should be one of number, spans or date at: {json.dumps(answer)}'
        )


def answer_json_to_string(answer_json):
    return json.dumps(answer_json_to_strings(answer_json))


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = ''.join(char for char in s if char not in exclude)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ' '.join(s.split())
    return s


def fuzzy_match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)

    if s1 == '' or s2 == '':
        return s1 == s2

    return s1 in s2 or s2 in s1


def drop_metric(sample: str, reference: list[str]) -> Tuple[float, float]:
    em_scores = []
    f1_scores = []
    for answer in reference:
        if answer.strip() != '':
            em, f1 = get_drop_metrics(sample, answer)
            em_scores.append(em)
            f1_scores.append(f1)
    return (max(em_scores), max(f1_scores))


class DropEval(Eval):

    def __init__(self,
                 num_examples: int | None = None,
                 train_samples_per_prompt: int = 3):
        self.seed = 42
        self._num_examples = num_examples
        self._train_samples_per_prompt = train_samples_per_prompt
        self.train_jsonl = (
            'https://openaipublic.blob.core.windows.net/simple-evals/drop_v0_train.jsonl.gz'
        )
        self.test_jsonl = (
            'https://openaipublic.blob.core.windows.net/simple-evals/drop_v0_dev.jsonl.gz'
        )
        with gzip.GzipFile(fileobj=common.url_to_fileobj(self.train_jsonl,
                                                         binary=True),
                           mode='rb') as f:
            self.train_samples = list(map(json.loads, f.readlines()))
        with gzip.GzipFile(fileobj=common.url_to_fileobj(self.test_jsonl,
                                                         binary=True),
                           mode='rb') as f:
            self.test_samples = list(map(json.loads, f.readlines()))
            if self._num_examples:
                self.test_samples = random.Random(self.seed).sample(
                    self.test_samples, self._num_examples)

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        rng = random.Random(self.seed)

        def fn(example: dict[str, str]):
            stuffing = rng.sample(self.train_samples,
                                  self._train_samples_per_prompt)

            # prompt = """TASK: Read the provided passage, then identify the correct answer to questions below."""
            prompt = """You will be asked to read a passage and answer a question. Some examples of passages and Q&A are provided below."""
            prompt += '\n\n# Examples'
            samples = stuffing + [example]
            for i, sample in enumerate(samples):
                is_test = i == len(stuffing)
                prompt += '\n# Your Task\n' if is_test else ''
                prompt += f"""
---
{sample["context"]} """

                a = sample['completion']
                correct_answers = sample['ref_text'].split('|')

                if not is_test:
                    prompt += a + '\n'
                else:
                    prompt += """\n
Think step by step, then write a line of the form "Answer: $ANSWER" at the end of your response.
                    """
                    prompt_messages = [
                        sampler._pack_message(content=prompt, role='user')
                    ]
                    sampler_response = sampler(prompt_messages)
                    response_text = sampler_response.response_text
                    actual_queried_prompt_messages = sampler_response.actual_queried_message_list
                    match = re.search(ANSWER_PATTERN, response_text)
                    extracted_answer = match.group(
                        1) if match else response_text
                    em_score, f1_score = drop_metric(extracted_answer,
                                                     correct_answers)
                    matches = [
                        fuzzy_match(extracted_answer, correct_answer)
                        for correct_answer in correct_answers
                    ]
                    extracted_answers = [
                        extracted_answer for i in range(len(correct_answers))
                        if matches[i]
                    ]
                    score = True in matches
                    html = common.jinja_env.from_string(HTML_JINJA).render(
                        prompt_messages=actual_queried_prompt_messages,
                        next_message=dict(content=extracted_answer,
                                          role='assistant'),
                        score=score,
                        correct_answer=correct_answers,
                        extracted_answer=extracted_answers,
                    )
                    convo = actual_queried_prompt_messages + [
                        dict(content=extracted_answer, role='assistant')
                    ]
                    return SingleEvalResult(
                        html=html,
                        score=score,
                        convo=convo,
                        metrics={
                            'em_score': em_score,
                            'f1_score': f1_score
                        },
                    )

        results = common.map_with_progress(fn, self.test_samples)
        return common.aggregate_results(results)
