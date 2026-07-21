import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Union

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import (DICT_POSTPROCESSORS, ICL_PROMPT_TEMPLATES,
                                  LOAD_DATASET)
from opencompass.utils.prompt import safe_format

from .base import BaseDataset

INSTRUCTION_TYPE_TO_ABBR = {
    'Question Correction': 'QC',
    'Intentional Textual Flaws': 'ITF',
    'Code without Comments': 'CC',
    'Counter-Conventional Formatting': 'CCF',
    'Deliberately Incorrect Answers': 'DIA',
    'Instructional Induction': 'II',
    'Mid-turn Instruction Modification': 'MIM',
    'Counterfactual Answering': 'CA',
}


@LOAD_DATASET.register_module()
class InverseIFEvalDataset(BaseDataset):
    """Load Inverse IFEval from HuggingFace.

    The official dataset stores the judge prompt template and judge system
    prompt per sample, so the loader keeps those fields for the second-stage
    LLM judge.
    """

    @staticmethod
    def load(path: str,
             name: Optional[str] = None,
             split: str = 'train',
             language: Optional[str] = None,
             instruction_type: Optional[Union[str, List[str]]] = None,
             max_samples: Optional[int] = None,
             **kwargs) -> Dataset:
        if name is None:
            raw_dataset = load_dataset(path, split=split, **kwargs)
        else:
            raw_dataset = load_dataset(path, name, split=split, **kwargs)

        if isinstance(raw_dataset, DatasetDict):
            raw_dataset = raw_dataset[split]

        if isinstance(instruction_type, str):
            instruction_types = {instruction_type}
        elif instruction_type is None:
            instruction_types = None
        else:
            instruction_types = set(instruction_type)

        records = []
        for item in raw_dataset:
            if language is not None and item.get('language') != language:
                continue
            if (instruction_types is not None and item.get('instruction_types')
                    not in instruction_types):
                continue

            records.append({
                'instruction_types':
                item.get('instruction_types', ''),
                'prompt':
                item.get('prompt', ''),
                'response_reference':
                item.get('response_reference', ''),
                'language':
                item.get('language', ''),
                'judge_prompt_template':
                item.get('judge_prompt_template', ''),
                'judge_system_prompt':
                item.get('judge_system_prompt', ''),
            })

            if max_samples is not None and len(records) >= max_samples:
                break

        return Dataset.from_list(records)


@ICL_PROMPT_TEMPLATES.register_module()
class InverseIFEvalJudgePromptTemplate:
    """Build per-sample judge messages for Inverse IFEval.

    The dataset's judge prompt template uses ``{response}`` for the model
    answer. OpenCompass injects model predictions as ``prediction`` before
    running ``GenericLLMEvaluator``, so this template bridges the two names.
    """

    prompt_type = 'raw_messages'
    sep_token = None
    ice_token = '</E>'

    def generate_item(self,
                      entry: Dict,
                      output_field=None,
                      output_field_replace_token='',
                      ice_field_replace_token='') -> List[Dict[str, str]]:
        values = dict(entry)
        values['response'] = entry.get('prediction', '')
        values['prediction'] = entry.get('prediction', '')

        judge_prompt_template = entry.get('judge_prompt_template', '')
        if not judge_prompt_template:
            judge_prompt_template = ('<题目>：\n{prompt}\n\n'
                                     '<标准答案>：\n{response_reference}\n\n'
                                     '<学生答案>：\n{response}')

        return [
            {
                'role': 'system',
                'content': entry.get('judge_system_prompt', ''),
            },
            {
                'role': 'user',
                'content': safe_format(judge_prompt_template, **values),
            },
        ]

    def generate_ice_item(self,
                          entry: Dict,
                          label=None) -> List[Dict[str, str]]:
        return self.generate_item(entry)

    def generate_label_prompt_item(self,
                                   entry: Dict,
                                   ice: str = '',
                                   label=None,
                                   remain_sep: bool = False
                                   ) -> List[Dict[str, str]]:
        return self.generate_item(entry)


def _extract_json_score(judgement: str) -> Optional[int]:
    code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```',
                             judgement,
                             flags=re.IGNORECASE)
    candidates = code_blocks + re.findall(r'\{[\s\S]*?answer_score[\s\S]*?\}',
                                          judgement)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
        score = parsed.get('answer_score')
        if score in (0, 1):
            return score
        if isinstance(score, str) and score.strip() in ('0', '1'):
            return int(score.strip())
    return None


def extract_inverse_ifeval_score(judgement: str) -> Optional[int]:
    if not isinstance(judgement, str) or not judgement.strip():
        return None

    json_score = _extract_json_score(judgement)
    if json_score is not None:
        return json_score

    patterns = [
        r'"answer_score"\s*:\s*"?([01])"?',
        r'answer_score\s*[:：]\s*"?([01])"?',
        r'【评分】\s*[:：]\s*([01])\s*分',
        r'评分\s*[:：]\s*([01])\s*分',
    ]
    for pattern in patterns:
        match = re.search(pattern, judgement)
        if match:
            return int(match.group(1))
    return None


def _empty_bucket() -> Dict[str, int]:
    return {'total': 0, 'correct': 0, 'parse_error': 0}


def _update_bucket(bucket: Dict[str, int], score: Optional[int]) -> None:
    bucket['total'] += 1
    if score == 1:
        bucket['correct'] += 1
    elif score is None:
        bucket['parse_error'] += 1


def _accuracy(bucket: Dict[str, int]) -> float:
    if bucket['total'] == 0:
        return 0.0
    return bucket['correct'] / bucket['total'] * 100


@DICT_POSTPROCESSORS.register_module()
def inverse_ifeval_judge_postprocess(output: dict,
                                     output_path: str,
                                     dataset=None) -> dict:
    overall = _empty_bucket()
    by_language = defaultdict(_empty_bucket)
    by_instruction_type = defaultdict(_empty_bucket)
    by_language_instruction_type = defaultdict(_empty_bucket)
    details = []

    test_set = dataset.test if dataset is not None else None

    for key in sorted(output, key=lambda x: int(x)):
        item = output[key]
        judgement = item.get('prediction', '')
        score = extract_inverse_ifeval_score(judgement)

        sample = test_set[int(key)] if test_set is not None else {}
        language = sample.get('language', '')
        instruction_type = sample.get('instruction_types', '')

        _update_bucket(overall, score)
        if language:
            _update_bucket(by_language[language], score)
        if instruction_type:
            _update_bucket(by_instruction_type[instruction_type], score)
        if language and instruction_type:
            _update_bucket(
                by_language_instruction_type[(language, instruction_type)],
                score)

        details.append({
            'idx':
            int(key),
            'prompt':
            sample.get('prompt', ''),
            'response_reference':
            sample.get('response_reference', ''),
            'model_response':
            sample.get('prediction', ''),
            'judge_response':
            judgement,
            'score':
            score,
            'correct':
            score == 1,
            'language':
            language,
            'instruction_types':
            instruction_type,
        })

    result = {
        'accuracy':
        _accuracy(overall),
        'correct_count':
        overall['correct'],
        'total':
        overall['total'],
        'parse_error_count':
        overall['parse_error'],
        'parse_error_rate': (overall['parse_error'] / overall['total'] *
                             100 if overall['total'] else 0.0),
        'details':
        details,
    }

    for language, bucket in sorted(by_language.items()):
        result[f'{language}_accuracy'] = _accuracy(bucket)
        result[f'{language}_count'] = bucket['total']

    for instruction_type, bucket in sorted(by_instruction_type.items()):
        abbr = INSTRUCTION_TYPE_TO_ABBR.get(instruction_type, instruction_type)
        result[f'{abbr}_accuracy'] = _accuracy(bucket)
        result[f'{abbr}_count'] = bucket['total']

    for (language, instruction_type), bucket in sorted(
            by_language_instruction_type.items()):
        abbr = INSTRUCTION_TYPE_TO_ABBR.get(instruction_type, instruction_type)
        result[f'{language}_{abbr}_accuracy'] = _accuracy(bucket)
        result[f'{language}_{abbr}_count'] = bucket['total']

    return result
