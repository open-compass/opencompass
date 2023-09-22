import json
import math
import random
import re
import time
from typing import List

import numpy as np
import requests

from opencompass.models import OpenAI

from .icl_base_evaluator import BaseEvaluator

DEFAULT_FAIL_WORDS = ('sorry', 'apologize', 'apology', 'unfortunately',
                      "couldn't")

CHECK_SOLVE_QUERY_PROMPT = '''\
Please check whether the answer solve the query or not.
Query:
{query}

Answer:
{answer}

Now give your judgment of JSON to `{func_name}`, remember do not be too strict.
'''

SELECT_BEST_ANSWER_PROMPT = '''\
For query {query}, you have the following answers in JSON format:
{answers}

I want you to select the best answer from the above answers and give the index of the answer of JSON to `{func_name}`. Now select the best answer.'''  # noqa: E501


def extract_answer(result: dict):
    """Extract answer from toolbench format."""
    final_answer = result['final_answer']
    try:
        final_answer = json.loads(final_answer)['final_answer']
    except Exception:
        pass

    next_step = result['answer_details']
    steps = []

    while len(next_step) > 0:
        step = next_step[-1]
        next_step = step['next']
        if step['role'] == 'tool':
            tool_type = re.findall(r"'name': '(.*?)'", step['message'])
            error = re.findall(r"{\"error\": \"([^\"]+)", step['message'])
            if len(tool_type) > 0:
                tool_type = tool_type[0]
                valid = 0
            else:
                tool_type = None
                valid = -2
            if tool_type == 'Finish':
                valid = 1
            if len(error) > 0:
                valid = -2
        elif step['role'] == 'assistant':
            tool_type = None
            valid = -2
        else:
            continue
        steps.append(
            dict(
                type=tool_type,
                args=None,
                result=None,
                thought=None,
                state=0,
                valid=valid,
            ))
    return final_answer, steps


class PassRateEvaluator(BaseEvaluator):
    """This Evaluator can determine whether pred refuses to execute the
    task."""

    def __init__(self, fail_words=DEFAULT_FAIL_WORDS) -> None:
        super().__init__()
        self.fail_words = fail_words

    def score(self, predictions: List, references: List = None) -> dict:
        results = []
        for pred in predictions:
            if pred and self.check_real_valid(pred):
                results.append(1)
            else:
                results.append(0)
        pass_rate = sum(results) / len(results) * 100
        return dict(pass_rate=pass_rate)

    def check_real_valid(self, answer):
        """Exclude response without real answer."""
        return not any(word in answer.lower() for word in self.fail_words)


class WinRateEvaluator(BaseEvaluator):
    # https://github.com/OpenBMB/ToolBench/blob/e18a30ed8f9afc131a7e313d0522c4371f030f31/toolbench/tooleval/evaluators/registered_cls/tooleval.py#L50
    """Follow `OpenAINormalizedEvaluator` in the `ToolBench`.

    The Evaluator will compare which call-tool process between `pred` and
    `reference` is better.

    1. Compare whether an answer can be extracted. The one that can extract an
       answer wins.
    2. If both can, then compare whether the answer is correct. The correct one
       wins.
    3. If both answers are correct, then compare the number of tool calls; the
       one with fewer calls wins. If the number of steps is the same, the one
       with the better-looking final answer wins.
    4. If both answers are incorrect, then consider factors such as whether the
       tool was successfully called and the variety of tools used.
    """

    def __init__(self,
                 model='gpt-3.5-turbo-16k',
                 temperature=0,
                 **kwargs) -> None:
        super().__init__()
        self.openai = OpenAI(path=model, temperature=temperature, **kwargs)

    def score(self, predictions: List, references: List, origin_prompt: List,
              steps: List):
        compare_list = []
        for query, ref, pred_answer, pred_steps in zip(origin_prompt,
                                                       references, predictions,
                                                       steps):
            ref_answer, ref_steps = extract_answer(ref)

            if bool(pred_answer) ^ bool(ref_answer):
                # Empty vs non-empty
                win = int(bool(pred_answer))
            else:
                pred_valid = bool(pred_answer) and self.check_solve_query(
                    query, pred_answer)
                ref_valid = bool(ref_answer) and self.check_solve_query(
                    query, ref_answer)

                if pred_valid and ref_valid:
                    # both answer success
                    if len(pred_steps) != len(ref_steps):
                        win = 1 if len(pred_steps) < len(ref_steps) else 0
                    else:
                        win = self.select_best_final_answer(
                            query, [ref_answer, pred_answer])
                elif not pred_valid and not ref_valid:
                    # both answer failed
                    win = self.compare_steps([ref_steps, pred_steps])
                else:
                    win = int(pred_valid)

            compare_list.append(win)

            pred_answer = pred_answer.replace('\n', '')
            ref_answer = ref_answer.replace('\n', '')
        return {'win_rate': sum(compare_list) / len(compare_list) * 100.}

    def check_solve_query(self, query: str, answer: str) -> bool:
        """Check whether the answer solved the query."""
        func_name = 'check_solve_query'
        return_key = 'is_solved'

        prompt = CHECK_SOLVE_QUERY_PROMPT.format(query=query,
                                                 answer=answer,
                                                 func_name=func_name)

        function = dict(
            name=func_name,
            description=('Check whether the given answer solve the given '
                         'query, return true or false'),
            parameters={
                'type': 'object',
                'properties': {
                    return_key: {
                        'type': 'boolean',
                        'description': 'true if solved and false if not'
                    }
                },
                'required': [return_key]
            })

        result = self._openai_function(
            prompt,
            max_out_len=100,
            functions=[function],
            function_call={'name': function['name']},
        )
        return bool(result[return_key])

    def select_best_final_answer(self, query: str, answers: list) -> int:
        """Select the best final answer from candidates."""
        func_name = 'select_best_final_answer'
        return_key = 'best_answer_index'

        is_reversed = random.random() > 0.5
        if is_reversed:
            answers = list(reversed(answers))
        prompt = SELECT_BEST_ANSWER_PROMPT.format(query=query,
                                                  answers=answers,
                                                  func_name=func_name)

        function = dict(
            name=func_name,
            description=('For given query, select the best answer in answers '
                         'list and return the index of the best answer'),
            parameters={
                'type': 'object',
                'properties': {
                    return_key: {
                        'type':
                        'number',
                        'description': ('The index of the best answer in the '
                                        'answer list, start from 0')
                    }
                },
                'required': [return_key]
            })

        result = self._openai_function(
            prompt,
            max_out_len=100,
            functions=[function],
            function_call={'name': function['name']},
        )
        if not is_reversed:
            return int(result[return_key])
        else:
            return len(answers) - int(result[return_key]) - 1

    def compare_steps(self, steps_list: list) -> int:
        """Compare results according to score when both answers are failed."""
        # calculate socre and return one with highest score
        scores = []
        for steps in steps_list:
            succeed_tool_calling = sum(step['valid'] == 0 for step in steps)
            used_tool_types = len(set(step['type'] for step in steps))
            score = succeed_tool_calling * 10 + used_tool_types * 5
            if len(steps) <= 0:
                score -= int(1e5)
            else:
                score += -5 * math.log(len(steps))
            scores.append(score)

        # return index of highest score
        scores = np.array(scores)
        highest_idx = np.where(scores == scores.max())[0].tolist()
        return random.choice(highest_idx)

    def _openai_function(self, msg: str, max_out_len: int, functions: dict,
                         function_call: dict, **kwargs) -> dict:
        openai = self.openai

        messages = [{'role': 'user', 'content': msg}]

        max_num_retries = 0
        while max_num_retries < openai.retry:
            openai.wait()

            if len(openai.invalid_keys) == len(openai.keys):
                raise RuntimeError('All keys have insufficient quota.')

            # find the next valid key
            while True:
                openai.key_ctr += 1
                if openai.key_ctr == len(openai.keys):
                    openai.key_ctr = 0

                if openai.keys[openai.key_ctr] not in openai.invalid_keys:
                    break

            key = openai.keys[openai.key_ctr]

            header = {
                'Authorization': f'Bearer {key}',
                'content-type': 'application/json',
            }

            if openai.orgs:
                openai.org_ctr += 1
                if openai.org_ctr == len(openai.orgs):
                    openai.org_ctr = 0
                header['OpenAI-Organization'] = openai.orgs[openai.org_ctr]

            try:
                data = dict(model=openai.path,
                            messages=messages,
                            max_tokens=max_out_len,
                            n=1,
                            stop=None,
                            temperature=openai.temperature,
                            functions=functions,
                            function_call=function_call,
                            **kwargs)
                raw_response = requests.post(openai.url,
                                             headers=header,
                                             data=json.dumps(data))
            except requests.ConnectionError:
                openai.logger.error('Got connection error, retrying...')
                continue
            try:
                response = raw_response.json()
            except requests.JSONDecodeError:
                openai.logger.error('JsonDecode error, got',
                                    str(raw_response.content))
                continue
            try:
                result = response['choices'][0]['message']['function_call'][
                    'arguments']
                return json.loads(result)
            except KeyError:
                if 'error' in response:
                    if response['error']['code'] == 'rate_limit_exceeded':
                        time.sleep(1)
                        continue
                    elif response['error']['code'] == 'insufficient_quota':
                        openai.invalid_keys.add(key)
                        openai.logger.warn(f'insufficient_quota key: {key}')
                        continue

                    openai.logger.error('Find error message in response: ',
                                        str(response['error']))
            max_num_retries += 1

        raise RuntimeError('Calling OpenAI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')
