import copy
import json
import os.path as osp
import re

from datasets import Dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from .base import BaseDataset


def get_number(options):
    result_string = ''
    for i, option in enumerate(options, start=ord('A')):
        result_string += f'{chr(i)}. {option}\n'
    return result_string


def get_circular_example(entry, id):
    """For given example, generate four circular examples."""
    # Only 4 options is supported for current circular eval.
    circular_patterns = ['ABCD', 'BCDA', 'CDAB', 'DABC']
    data = []
    for c in circular_patterns:
        line = copy.deepcopy(entry)
        options = []
        for i in range(4):
            options.append(line['options'][ord(c[i]) - ord('A')])
        line['options'] = options
        line['answer'] = {
            c[0]: 'A',
            c[1]: 'B',
            c[2]: 'C',
            c[3]: 'D'
        }[line['answer']]
        line['answer'] = str(id) + '--' + line['answer'] + '--' + c
        line['question'] = line['question'].strip() + '\n' + get_number(
            line['options'])
        data.append(line)

    return data


@LOAD_DATASET.register_module()
class MathBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, with_circular: bool = True):
        """MathBenth Dataset.

        Args:
            path (str): Path of the mathbench dataset.
            name (str): Name of the target subset.
            with_circular (bool): Whether to create circular dataset for
                single choice question. Defaults to True.
        """
        path = get_data_path(path, local_mode=True)
        data = []
        filename = osp.join(path, f'{name}.jsonl')
        with open(filename, 'r', encoding='utf-8') as infile:
            for id, line in enumerate(infile):
                entry = json.loads(line)
                if 'cloze' in name:
                    data.append({
                        'question': entry['question'].strip(),
                        'answer': entry['answer'].strip()
                    })
                else:
                    if with_circular:
                        data.extend(get_circular_example(entry, id))
                    else:
                        question = entry['question'].strip(
                        ) + '\n' + get_number(entry['options'])
                        info = {
                            'question': question,
                            'answer': entry['answer'].strip()
                        }
                        # # For PPL evaluation
                        # for i in range(4):
                        #     info[chr(ord('A') +
                        #              i)] = entry['options'][i].strip()
                        data.append(info)

        dataset = Dataset.from_list(data)
        return dataset


@TEXT_POSTPROCESSORS.register_module()
def mathbench_postprocess(text: str, name: str) -> str:
    split = False
    ans = text
    if '_cn' in name:
        ans_line = ans.split('答案是')
    else:
        ans_line = ans.split('The answer is')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
        split = True

    output = re.sub(r'(\d),(\d)', r'\1\2', ans)
    numbers = re.findall(r'-?\d*\.?/?\d+|\d+', output)

    if numbers:
        return numbers[0] if split else numbers[-1]

    return ans


@LOAD_DATASET.register_module()
class MathBenchBuggyDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, with_circular: bool = True):
        data = []
        filename = osp.join(path, f'{name}.jsonl')
        with open(filename, 'r', encoding='utf-8') as infile:
            for id, line in enumerate(infile):
                entry = json.loads(line)
                if 'cloze' in name:
                    data.append({
                        'question': entry['question'].strip(),
                        'answer': entry['answer'].strip()
                    })
                else:
                    if with_circular:
                        data.extend(get_circular_example(entry, id))
                    else:
                        question = entry['question'].strip(
                        ) + '\n' + get_number(entry['options'])
                        info = {
                            'question': question,
                            'answer': entry['answer'].strip()
                        }
                        # For PPL evaluation
                        for i in range(4):
                            info[chr(ord('A') +
                                     i)] = entry['options'][i].strip()
                        data.append(info)

        if 'cloze' not in name:
            data = data[:(len(data) // 4 + 7) // 8 * 8]
        dataset = Dataset.from_list(data)
        return dataset


import collections

from ..openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from ..registry import ICL_EVALUATORS


def first_option_postprocess(text: str, options: str, cushion=True) -> str:
    """Find first valid option for text."""

    # yapf: disable
    # flake8: noqa: W605
    patterns = [
        f'答案是?\s*([{options}])',
        f'答案是?\s*：\s*([{options}])',
        f'答案是?\s*:\s*([{options}])',
        f'答案应该?是\s*([{options}])',
        f'答案应该?选\s*([{options}])',
        f'答案为\s*([{options}])',
        f'答案选\s*([{options}])',
        f'选择?\s*([{options}])',
        f'故选?\s*([{options}])'
        f'只有选?项?\s?([{options}])\s?是?对',
        f'只有选?项?\s?([{options}])\s?是?错',
        f'只有选?项?\s?([{options}])\s?不?正确',
        f'只有选?项?\s?([{options}])\s?错误',
        f'说法不?对选?项?的?是\s?([{options}])',
        f'说法不?正确选?项?的?是\s?([{options}])',
        f'说法错误选?项?的?是\s?([{options}])',
        f'([{options}])\s?是正确的',
        f'([{options}])\s?是正确答案',
        f'选项\s?([{options}])\s?正确',
        f'所以答\s?([{options}])',
        f'所以\s?([{options}][.。$]?$)',
        f'所有\s?([{options}][.。$]?$)',
        f'[\s，：:,]([{options}])[。，,\.]?$',
        f'[\s，,：:][故即]([{options}])[。\.]?$',
        f'[\s，,：:]因此([{options}])[。\.]?$',
        f'[是为。]\s?([{options}])[。\.]?$',
        f'因此\s?([{options}])[。\.]?$',
        f'显然\s?([{options}])[。\.]?$',
        f'回答[\s：:]\s?([{options}])',
        f'Answer[\s：:]\s?([{options}])',
        f'答案是\s?(\S+)(?:。|$)',
        f'答案应该是\s?(\S+)(?:。|$)',
        f'答案为\s?(\S+)(?:。|$)',
        f'[Tt]he answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
    ]
    cushion_patterns = [
        f'^选项\s?([{options}])',
        f'^([{options}])\s?选?项',
        # f'[\s|^]([{options}])[\s。，,：:\.$]',
        f'[\s|^]([{options}])[。，,：:\.$]',
        f'1.\s?([{options}])[.。$]?$',
        f'([{options}]):',
        f'([{options}])',
    ]
    # flake8: noqa
    # yapf: enable
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i, pattern
    if cushion:
        for pattern in cushion_patterns:
            outputs = []
            current_text = text
            while True:
                match = re.search(pattern, current_text, re.DOTALL)
                if match:
                    outputs.append(match.group(0))
                    current_text = current_text[match.end():]
                else:
                    break
            # if len(outputs) >= 2:
            #     from IPython import embed; embed(); exit()
            if outputs:
                outputs = outputs[-1]
                for i in options:
                    if i in outputs:
                        return i, pattern
    return '', None


def remove_invisible_chars(text: str) -> str:
    """Remove invisible characters."""
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'\u200b', '', text)
    return text


@ICL_EVALUATORS.register_module()
class MathBenchCircularEvaluator(BaseEvaluator):
    """Robust circular evaluator for multi-choice questions."""

    def __init__(self) -> None:
        super().__init__()
        self.cp4 = ['ABCD', 'BCDA', 'CDAB', 'DABC']
        self.cp1 = ['ABCD']

    def score(self, predictions, references, test_set):
        """Calculate the accuracy of predictions.

        Args:
            predictions (list): List of predictions.
            references (list): List of references.

        Returns:
            dict: A dict of evaluation results.
        """
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        extract_details = {}
        extracted_predictions = []
        for index, p in enumerate(predictions):
            extracted_p = None
            matched_pattern = None
            if '\\boxed' in p:
                match = re.findall(r'\\boxed\{(.*)\}', p)
                if match:
                    for m in match:
                        for j in range(4):
                            m = remove_invisible_chars(m)
                            o = remove_invisible_chars(
                                test_set[index]['options'][j])
                            if m == o:
                                extracted_p = chr(ord('A') + j)
                                matched_pattern = 'boxed_answer'
                                break
                        else:
                            if m in ['A', 'B', 'C', 'D']:
                                extracted_p = m
                                matched_pattern = 'boxed_ABCD'
                            else:
                                continue
                        break
            if extracted_p is None:
                extracted_p, matched_pattern = first_option_postprocess(
                    p, 'ABCD')
            extracted_predictions.append(extracted_p)
            extract_details[str(index)] = {
                'question': test_set[index]['question'],
                'options': test_set[index]['options'],
                'origin_pred': p,
                'extracted_pred': extracted_p,
                'matched_pattern': matched_pattern,
                'ref': references[index],
            }
        predictions = extracted_predictions

        results = {}
        results.update({'acc_4': 0, 'acc_1': 0})
        # Accuracy for patterns with no circular shift / 4 circular shifts
        for index, (pred, reference) in enumerate(zip(predictions,
                                                      references)):
            _, ref, circular_pattern = reference.split('--')
            extract_details[str(index)]['is_correct'] = pred == ref
            if circular_pattern in self.cp4:
                results['acc_4'] += 1 if pred == ref else 0
            if circular_pattern in self.cp1:
                results['acc_1'] += 1 if pred == ref else 0
        for k in ['acc_4', 'acc_1']:
            results[k] = results[k] / len(predictions) * 4 / int(
                k.split('_')[-1]) * 100

        # Accuracy for patterns with no circular shift / 4 circular shifts
        details = {4: {}, 1: {}}
        for pred, reference in zip(predictions, references):
            index, ref, circular_pattern = reference.split('--')
            if index not in details[4]:
                details[4][index] = []
                details[1][index] = []
            if circular_pattern in self.cp4:
                details[4][index].append(True if pred == ref else False)
            if circular_pattern in self.cp1:
                details[1][index].append(True if pred == ref else False)
        # Calculate accuracy for having at least j correct out of i total
        for i in [1, 4]:
            for j in range(0, i + 1):
                count, total = 0, 0
                for index in details[i]:
                    if sum(details[i][index]) >= j:
                        count += 1
                    total += 1
                results[f'more_{i}_{j}'] = count / total * 100
        # Consider fully correct as correct
        for i in [1, 4]:
            results[f'perf_{i}'] = results[f'more_{i}_{i}']

        # Calculate voting accuracy
        voting = {'vote_4': {}, 'vote_1': {}}
        refs = {}
        for pred, reference in zip(predictions, references):
            index, ref, circular_pattern = reference.split('--')
            c = circular_pattern
            back_map = {'A': c[0], 'B': c[1], 'C': c[2], 'D': c[3]}
            ref = back_map[ref]
            if pred not in ['A', 'B', 'C', 'D']:
                pred = '-'
            else:
                pred = back_map[pred]
            if index not in voting['vote_4']:
                voting['vote_4'][index] = collections.Counter()
                voting['vote_1'][index] = collections.Counter()
                refs[index] = ref

            if c in self.cp4:
                voting['vote_4'][index][pred] += 1
            if c in self.cp1:
                voting['vote_1'][index][pred] += 1
        for k in ['vote_4', 'vote_1']:
            voting_count = 0
            for index in voting[k]:
                if refs[index] == voting[k][index].most_common(1)[0][0]:
                    voting_count += 1
            results[k] = voting_count / len(voting[k]) * 100

        # Calculate the frequency of ABCD in model predictions
        prior_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, '-': 0}
        for pred, reference in zip(predictions, references):
            if pred in ['A', 'B', 'C', 'D']:
                prior_counts[pred] += 1
            else:
                prior_counts['-'] += 1
        for k in ['A', 'B', 'C', 'D', '-']:
            results[f'prior_{k}'] = prior_counts[k] / len(predictions) * 100

        results['details'] = extract_details
        return results
