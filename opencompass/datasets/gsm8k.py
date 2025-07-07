import json
import os
import re
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class GSM8KDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(dataset_name=path)
        else:
            datasets = {}
            for split in ['train', 'test']:
                split_path = os.path.join(path, split + '.jsonl')
                dataset = []
                if split == "train":
                    with open(split_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = json.loads(line.strip())    
                            cot_ans = line['answer'].split('#### ')[0]
                            lable = line['answer'].split('#### ')[1]
                            line["lable"] = lable.replace(',', '')   
                            line["answer"] = cot_ans
                            dataset.append(line)
                    datasets[split] = Dataset.from_list(dataset)
                else:  
                    with open(split_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = json.loads(line.strip())    
                            dataset.append(line)
                    datasets[split] = Dataset.from_list(dataset)
            dataset = DatasetDict(datasets)
        return dataset


@TEXT_POSTPROCESSORS.register_module('gsm8k_dataset')
def gsm8k_dataset_postprocess(text: str) -> str:
    return text.split('#### ')[1].replace(',', '')
def last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def remove_boxed(s):
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except Exception:
        return None
    
def extract_boxed_answer(pred_str, strip_double_curly_brace=False):
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        return None
    answer = remove_boxed(boxed_str)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match('^\{(.*)\}$', answer)  # noqa: W605
        if match:
            answer = match.group(1)
    return answer

@TEXT_POSTPROCESSORS.register_module('gsm8k')


def gsm8k_postprocess_original(text: str) -> str:
    # 优先抽取boxed中的数字
    text = text.replace(',', '')
    cand_ans = extract_boxed_answer(text, strip_double_curly_brace=True)
    if cand_ans:
        numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', cand_ans)
        if not numbers:
            return 'NULL'
        return numbers[-1]
    text = text.split('Question:')[0]
    numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
    if not numbers:
        return 'NULL'
    return numbers[-1]

def gsm8k_postprocess(text: str) -> str:
    # 优先抽取boxed中的数字
    text = text.replace(',', '')
    cand_ans = extract_boxed_answer(text, strip_double_curly_brace=True)
    if cand_ans:
        numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', cand_ans)
        if not numbers:
            return 'NULL'
        return numbers[-1]
    text = text.split('Question:')[0]
    numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
    if not numbers:
        return 'NULL'
    return numbers[-1]

class Gsm8kEvaluator(BaseEvaluator):

    def is_equal(self, pred, refer):
        try:
            if pred == refer or abs(float(pred) - int(refer)) < 1e-6:
                return True
        except Exception:
            pass
        return False

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        correct = 0
        count = 0
        details = []
        for i, j in zip(predictions, references):
            detail = {'pred': i, 'answer': j, 'correct': False}
            count += 1
            if self.is_equal(i, j):
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result


class Gsm8kAgentEvaluator(BaseEvaluator):
    """Gsm8k agent evaluator for soft condition.

    Args:
        action (str): Action for catching internal prediction.
            Defaults to `PythonInterpreter`.
    """

    def __init__(self, action: str = 'PythonInterpreter'):
        self.action = action

    def is_equal(self, pred, refer):
        try:
            if pred == refer or abs(float(pred) - int(refer)) < 1e-6:
                return True
        except Exception:
            pass
        return False

    def soft_equal(self, pred, refer, step):
        try:
            soft_pred = step['result']['text']
            if abs(float(soft_pred) - int(refer)) < 1e-6:
                return True
        except Exception:
            # result might not exists
            # text cannot convert to float
            pass
        return False

    def get_action(self, step):
        for s in step[::-1]:
            if s['type'] == self.action:
                return s

    def score(self, predictions, references, steps):
        """Calculate accuracy."""
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        row_reasoning_scope = 0
        action_scope = 0
        code_scope = 0
        reasoning_scope = 0
        final_scope = 0
        total = len(references)
        for pred, refer, step in zip(predictions, references, steps):
            # if final answer right
            if self.is_equal(pred, refer):
                if self.get_action(step):
                    final_scope += 1
                else:
                    row_reasoning_scope += 1
            else:
                s = self.get_action(step)
                if s:
                    action_scope += 1
                    if not s['errmsg']:
                        code_scope += 1
                        # whether action result is correct
                        reasoning_scope += self.soft_equal(pred, refer, s)

        result = dict(
            follow_acc=100 * (row_reasoning_scope + final_scope) / total,
            reasoning_acc=100 *
            (reasoning_scope + final_scope + row_reasoning_scope) / total,
            code_acc=100 * (code_scope + final_scope) / total,
            action_pct=100 * (action_scope + final_scope) / total,
        )
        return result
