import hashlib
import json
import os
import re
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from typing import Any, Callable, Literal

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset
from .sampler.chat_completion_sampler import ChatCompletionSampler
from .types import EvalResult, MessageList, SingleEvalResult


def map_with_progress(
    f: Callable,
    xs: list[Any],
    num_threads: int = os.cpu_count() or 10,
    pbar: bool = True,
):
    """Apply f to each element of xs, using a ThreadPool, and show progress."""
    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x

    if os.getenv('debug'):
        return list(map(f, pbar_fn(xs, total=len(xs))))
    else:
        with ThreadPool(min(num_threads, len(xs))) as pool:
            return list(pbar_fn(pool.imap(f, xs), total=len(xs)))


OPENAI_SYSTEM_MESSAGE_API = 'You are a helpful assistant.'
max_tokens_judge = 2048

GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.
# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
"explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
"criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
"explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
"criteria_met": false
}
```

As another example, if the criteria says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
"explanation": "The response is concise, so the criteria was not met.",
"criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()  # noqa: E501


class RubricItem:

    def __init__(self, criterion: str, points: float, tags: list[str]):
        self.criterion = criterion
        self.points = points
        self.tags = tags

    def __str__(self):
        return f'[{self.points}] {self.criterion}'

    def to_dict(self):
        return {
            'criterion': self.criterion,
            'points': self.points,
            'tags': self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            criterion=d['criterion'],
            points=d['points'],
            tags=d['tags'],
        )


def _parse(item):
    prompt = item['prompt'] + [dict(role='assistant', content='')]
    item['prompt_trans'] = prompt
    return item


def parse_json_to_dict(json_string: str) -> dict:
    json_cleaned = re.sub(r'^```json\s*|\s*```$', '',
                          json_string.strip())  # noqa: W291, E501
    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        print(f'JSON decoding failed: {e}')
        return {}


def calculate_score(
        rubric_items: list[RubricItem],
        grading_response_list: list[dict]) -> float | None:  # noqa: E501
    total_possible_points = sum(rubric_item.points
                                for rubric_item in rubric_items
                                if rubric_item.points > 0  # noqa: E501
                                )
    if total_possible_points == 0:
        # should not happen for overall score, but may happen for tags
        return None

    achieved_points = sum(rubric_item.points
                          for rubric_item, grading_response in zip(
                              rubric_items, grading_response_list, strict=True)
                          if grading_response['criteria_met'])
    overall_score = achieved_points / total_possible_points
    return overall_score


def get_usage_dict(response_usage) -> dict[str, int | None]:
    if response_usage is None:
        return {
            'input_tokens': None,
            'input_cached_tokens': None,
            'output_tokens': None,
            'output_reasoning_tokens': None,
            'total_tokens': None,
        }

    try:
        input_tokens = response_usage.input_tokens
        input_tokens_details = response_usage.input_tokens_details
        output_tokens = response_usage.output_tokens
        output_tokens_details = response_usage.output_tokens_details
        total_tokens = response_usage.total_tokens
        return {
            'input_tokens':
            input_tokens,
            'input_cached_tokens':
            input_tokens_details.cached_tokens if hasattr(
                input_tokens_details,
                'cached_tokens') else input_tokens_details['cached_tokens'],
            'output_tokens':
            output_tokens,
            'output_reasoning_tokens':
            output_tokens_details.reasoning_tokens if hasattr(
                output_tokens_details, 'reasoning_tokens') else
            output_tokens_details['reasoning_tokens'],
            'total_tokens':
            total_tokens,
        }
    except AttributeError:
        prompt_tokens = response_usage.prompt_tokens
        prompt_tokens_details = response_usage.prompt_tokens_details
        completion_tokens = response_usage.completion_tokens
        completion_tokens_details = response_usage.completion_tokens_details  # noqa: E501
        total_tokens = response_usage.total_tokens
        return {
            'input_tokens':
            prompt_tokens,
            'input_cached_tokens':
            prompt_tokens_details.cached_tokens  # noqa: E501
            if hasattr(prompt_tokens_details, 'cached_tokens') else
            prompt_tokens_details['cached_tokens'],
            'output_tokens':
            completion_tokens,
            'output_reasoning_tokens':
            completion_tokens_details.reasoning_tokens  # noqa: E501
            if hasattr(completion_tokens_details, 'reasoning_tokens') else
            completion_tokens_details['reasoning_tokens'],
            'total_tokens':
            total_tokens,
        }


def _compute_clipped_stats(
    values: list,
    stat: str,
):
    """Computes the mean (clipped to [0, 1]), bootstrap std for that mean, and
    n_samples for final HealthBench scoring."""
    if stat == 'mean':
        return np.clip(np.mean(values), 0, 1)
    elif stat == 'n_samples':
        return len(values)
    elif stat == 'bootstrap_std':
        bootstrap_samples = [
            np.random.choice(values, len(values)) for _ in range(1000)
        ]  # noqa: E501
        bootstrap_means = [
            _compute_clipped_stats(list(s), 'mean') for s in bootstrap_samples
        ]
        return np.std(bootstrap_means)
    else:
        raise ValueError(f'Unknown {stat =}')


def _aggregate_get_clipped_mean(
    single_eval_results: list[SingleEvalResult],
) -> EvalResult:  # noqa: E501, E125
    name2values = defaultdict(list)
    htmls = []
    convos = []
    metadata = []
    for single_eval_result in single_eval_results:
        for name, value in single_eval_result.metrics.items():
            name2values[name].append(value)
        if single_eval_result.score is not None:
            name2values['score'].append(single_eval_result.score)
        htmls.append(single_eval_result.html)
        convos.append(single_eval_result.convo)
        metadata.append(single_eval_result.example_level_metadata)
    final_metrics = {}
    for name, values in name2values.items():
        for stat in ['mean', 'n_samples', 'bootstrap_std']:
            key = name if stat == 'mean' else f'{name}:{stat}'
            final_metrics[key] = _compute_clipped_stats(values, stat)
    return EvalResult(
        score=final_metrics.pop('score', None),
        metrics=final_metrics,
        htmls=htmls,
        convos=convos,
        metadata={'example_level_metadata': metadata},
    )


@LOAD_DATASET.register_module()
class HealthBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, **kwargs):
        subset = kwargs.get('subset')
        if subset == '':
            data_files = {'test': '2025-05-07-06-14-12_oss_eval.jsonl'}
        elif subset == 'hard':
            data_files = {'test': 'hard_2025-05-08-21-00-10.jsonl'}
        elif subset == 'consensus':
            data_files = {'test': 'consensus_2025-05-09-20-00-46.jsonl'}
        else:
            raise Exception(f'Unrecognized subset type: {subset}')
        dataset = load_dataset(path, data_files=data_files, split='test')
        # dataset = dataset.select(range(2))
        dataset = dataset.map(lambda item: _parse(item))
        return dataset


class HealthBenchEvaluator(BaseEvaluator):
    """only consider the model completion mode, not physician mode / reference
    mode."""

    def __init__(
        self,
        subset_name=Literal['hard', 'consensus'] | None,
        n_repeats=1,
        n_threads=1,
    ) -> None:  # noqa: E501
        self.n_repeats = n_repeats
        self.n_threads = n_threads
        self.subset_name = subset_name
        self.grader_model = ChatCompletionSampler(
            model=os.environ['OC_JUDGE_MODEL'],
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        )  # noqa: E501

    def grade_sample(
        self,
        prompt: list[dict[str, str]],
        response_text: str,
        example_tags: list[str],
        rubric_items: list[RubricItem],
    ) -> tuple[dict, str, list[dict]]:  # noqa: E501
        # construct and grade the sample
        convo_with_response = prompt + [
            dict(content=response_text, role='assistant')
        ]  # noqa: E501

        def grade_rubric_item(rubric_item: RubricItem) -> dict:
            convo_str = '\n\n'.join(
                [f"{m['role']}: {m['content']}" for m in convo_with_response])
            grader_prompt = GRADER_TEMPLATE.replace('<<conversation>>',
                                                    convo_str).replace(
                                                        '<<rubric_item>>',
                                                        str(rubric_item))
            messages: MessageList = [dict(content=grader_prompt, role='user')]
            while True:
                sampler_response = self.grader_model(messages)
                grading_response = sampler_response.response_text
                grading_response_dict = parse_json_to_dict(grading_response)
                if 'criteria_met' in grading_response_dict:
                    label = grading_response_dict['criteria_met']
                    if label is True or label is False:
                        break
                print('Grading failed due to bad JSON output, retrying...')
            return grading_response_dict

        grading_response_list = map_with_progress(
            grade_rubric_item,
            rubric_items,
            pbar=False,
        )

        # compute the overall score
        overall_score = calculate_score(rubric_items, grading_response_list)
        assert overall_score is not None
        metrics = {
            'overall_score': overall_score,
        }

        # compute scores for example-level tags)
        example_tag_scores = {tag: overall_score for tag in example_tags}
        assert len(example_tag_scores) == len(example_tags)  # No duplicates.
        metrics.update(example_tag_scores)

        # compute scores for rubric-level tags
        rubric_tag_items_grades = defaultdict(list)
        for rubric_item, grading_response in zip(
                rubric_items, grading_response_list):  # noqa: E501
            curr_item_tags = set()  # Ensure no duplicates in a rubric item.
            for tag in rubric_item.tags:
                rubric_tag_items_grades[tag].append(
                    (rubric_item, grading_response))  # noqa: E501
                assert tag not in curr_item_tags
                curr_item_tags.add(tag)

        rubric_tag_scores = {}
        for tag, items_grades in rubric_tag_items_grades.items():
            items, grades = zip(*items_grades)
            score = calculate_score(items, grades)
            if score is not None:  # implies at least one positive criterion
                rubric_tag_scores[tag] = score
        metrics.update(rubric_tag_scores)

        # construct the list of explanations and grades
        rubric_items_with_grades = []
        readable_explanation_list = []
        for rubric_item, grading_response in zip(
                rubric_items, grading_response_list):  # noqa: E501
            explanation = grading_response.get(
                'explanation', 'No explanation provided')  # noqa: E501
            criteria_met = grading_response['criteria_met']
            readable_explanation = (f'[{criteria_met}] {rubric_item}\
                        Explanation: {explanation}')
            readable_explanation_list.append(readable_explanation)
            rubric_items_with_grades.append({
                **rubric_item.to_dict(),
                'criteria_met':
                criteria_met,
                'explanation':
                explanation,
            })

        readable_explanation_list.sort(key=lambda x: x.startswith('[False]'),
                                       reverse=True)
        readable_explanation_str = '\n\n'.join(readable_explanation_list)
        readable_explanation_str = f'\n\n{readable_explanation_str}'

        return metrics, readable_explanation_str, rubric_items_with_grades

    def score(self, predictions, references, test_set):
        results = []
        if len(predictions) != len(references):
            return {
                'error': 'preds and refrs have different length'
            }  # noqa: W291, E501
        for idx, (i, j) in enumerate(zip(predictions, references)):
            response_usage = None
            actual_queried_prompt_messages = test_set[idx]['prompt']
            response_text = i
            row = test_set[idx]  # noqa: W291
            metrics, readable_explanation_str, rubric_items_with_grades = (
                self.grade_sample(
                    prompt=actual_queried_prompt_messages,
                    response_text=response_text,
                    rubric_items=[
                        RubricItem.from_dict(d) for d in row['rubrics']
                    ],  # noqa: E501
                    example_tags=row['example_tags'],
                ))

            score = metrics['overall_score']
            convo = actual_queried_prompt_messages + [
                dict(content=response_text, role='assistant')
            ]
            results.append(
                SingleEvalResult(
                    html=None,
                    score=score,
                    convo=convo,
                    metrics=metrics,
                    example_level_metadata={
                        'score':
                        score,
                        'usage':
                        get_usage_dict(response_usage),
                        'rubric_items':
                        rubric_items_with_grades,
                        'prompt':
                        actual_queried_prompt_messages,
                        'completion':
                        [dict(content=response_text,
                              role='assistant')],  # noqa: E501
                        'prompt_id':
                        row['prompt_id'],
                        'completion_id':
                        hashlib.sha256(
                            (row['prompt_id'] +
                             response_text).encode('utf-8')).hexdigest(),
                    },
                ))
        results = _aggregate_get_clipped_mean(results)
        assert results.metrics is not None
        metrics = results.metrics | {'score': results.score}
        metrics = dict(sorted(metrics.items()))
        acc = metrics.get('f1_score', metrics.get('score', None))
        return {'accuracy': acc}
