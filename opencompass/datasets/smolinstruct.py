# flake8: noqa: W605
import re
from collections import defaultdict

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from nltk.translate.meteor_score import meteor_score

from opencompass.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from opencompass.registry import (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)
from opencompass.utils import get_logger

from .base import BaseDataset


@LOAD_DATASET.register_module()
class SmolInstructDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        raw_dataset = load_dataset(path, trust_remote_code=True)
        for split in ['validation', 'test']:
            raw_data = []
            for data in raw_dataset[split]:
                if data['task'] == name:
                    raw_data.append(data)
            dataset[split] = Dataset.from_list(raw_data)
        return dataset


def extract_chemical_data(text):
    other_patterns = [
        'reactants and reagents are:\n```\n', 'reactants and reagents:\n```\n',
        'Reactants and Reagents:**\n```\n',
        'Reactants and Reagents SMILES:**\n```\n',
        'Reactants and Reagents:**  \n`'
    ]

    pattern = re.compile(r'<(MOLFORMULA|SMILES|IUPAC)>(.*?)</\1>', re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        for other_pattern in other_patterns:
            if other_pattern in text:
                text = text.split(other_pattern)[-1].split('\n')[0]
                break
        return [text]
    return [match[1].strip() for match in matches]


def parse_molecule(molecular_formula):
    valid = re.match('([A-Za-z]\d*)+([\+\-]\d*)*$', molecular_formula)
    if valid is None:
        raise ValueError("Molecular formula \"%s\" is not valid." %
                         molecular_formula)

    stack = [defaultdict(int)]

    def _parse_formula(formula, _stack):

        # Set remainder equal to 'None'
        r = None

        # Regular expression matching for each of the three cases:
        atom = re.match(r'([A-Z][a-z]?)(\d+)?', formula)
        opening = re.match(r'[\(\[\{]', formula)
        closing = re.match(r'[\)\]\}](\d+)?', formula)

        # If atom is identified:
        if atom:
            r = formula[len(atom.group()):]
            _stack[-1][atom.group(1)] += int(atom.group(2) or 1)

        # If opening brackets encountered:
        elif opening:
            r = formula[len(
                opening.group()
            ):]  # this sets the remainder equal to everything after the opening brackets
            _stack.append(defaultdict(int))

            # If closing brackets encountered:
        elif closing:
            r = formula[len(
                closing.group()
            ):]  # this sets the remainder equal to everything after the closing brackets
            for k, v in _stack.pop().items():
                _stack[-1][k] += v * int(
                    closing.group(1)
                    or 1)  # v times amount of molecule k, depending on nesting

        # If anything remains, process remainders recursively as nested formulas:
        if r:
            _parse_formula(r, _stack)

        return dict(_stack[0])

    result = _parse_formula(molecular_formula, stack)

    charge = re.search('[\+\-]\d*', molecular_formula)
    if charge is not None:
        charge_str = charge.group()
        charge_type = charge_str[0]
        if len(charge_str) == 1:
            charge_num = 1
        else:
            charge_num = int(charge_str[1:])
        result[charge_type] = charge_num

    return result


def calculate_single_element_match_for_list(predictions, references):
    # 抽取SMILES里的化学式
    predictions = [
        extract_chemical_data(prediction) for prediction in predictions
    ]
    references = [extract_chemical_data(reference) for reference in references]

    ele_match_labels = []
    ele_invalid_labels = []
    details = []
    for pred_formula, gold_formula in zip(predictions, references):
        gold_formula = gold_formula[-1]
        if pred_formula:
            pred_formula = pred_formula[-1]
        detail = {'pred': [pred_formula], 'answer': gold_formula}
        if not pred_formula or not pred_formula:
            ele_invalid_labels.append(False)
            ele_match_labels.append(False)
            detail['score'] = [False]
            details.append(detail)
            continue
        try:
            pred_ele = parse_molecule(pred_formula)
        except KeyboardInterrupt:
            raise
        except:
            # print(pred_formula)
            # print('=====')
            ele_invalid_labels.append(True)
            ele_match_labels.append(False)
            detail['score'] = [False]
            details.append(detail)
            continue
        ele_invalid_labels.append(False)
        ele_match = False
        gold_ele = parse_molecule(gold_formula)
        if pred_ele == gold_ele:
            ele_match = True
        ele_match_labels.append(ele_match)
        detail['score'] = [ele_match]
        details.append(detail)

    score = sum(ele_match_labels) / len(predictions) * 100
    valid_score = 100 - sum(ele_invalid_labels) / len(predictions) * 100

    return {'score': score, 'valid_score': valid_score, 'details': details}


def calculate_single_element_match(predictions, references):
    # 抽取SMILES里的化学式
    predictions = [
        extract_chemical_data(prediction) for prediction in predictions
    ]
    references = [extract_chemical_data(reference) for reference in references]

    ele_match_labels = []
    ele_invalid_labels = []
    details = []
    for pred_formula, gold_formula in zip(predictions, references):
        gold_formula = gold_formula[-1]
        if pred_formula:
            pred_formula = pred_formula[-1]
        detail = {'pred': pred_formula, 'answer': gold_formula}
        if not pred_formula or not pred_formula:
            ele_invalid_labels.append(False)
            ele_match_labels.append(False)
            detail['score'] = False
            details.append(detail)
            continue
        try:
            pred_ele = parse_molecule(pred_formula)
        except KeyboardInterrupt:
            raise
        except:
            # print(pred_formula)
            # print('=====')
            ele_invalid_labels.append(True)
            ele_match_labels.append(False)
            detail['score'] = False
            details.append(detail)
            continue
        ele_invalid_labels.append(False)
        ele_match = False
        gold_ele = parse_molecule(gold_formula)
        if pred_ele == gold_ele:
            ele_match = True
        ele_match_labels.append(ele_match)
        detail['score'] = ele_match
        details.append(detail)

    score = sum(ele_match_labels) / len(predictions) * 100
    valid_score = 100 - sum(ele_invalid_labels) / len(predictions) * 100

    return {'score': score, 'valid_score': valid_score, 'details': details}


@ICL_EVALUATORS.register_module()
class NCElementMatchEvaluator(BaseEvaluator):
    """Element match evaluator for name conversion."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        print('len(predictions):', len(predictions))
        print('len(references):', len(references))
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        # topk的prediction，要拆开
        if isinstance(predictions[0], str):
            return calculate_single_element_match(predictions, references)
        else:
            num_k = len(predictions[0])
            scores = []
            for i in range(num_k):
                pred = [prediction[i] for prediction in predictions]
                ref = references
                score = calculate_single_element_match_for_list(pred, ref)
                scores.append(score)
            # 按照instance合并成一个完整的dict
            final_details = scores[0]['details']
            final_scores = [scores[0]['score']]
            final_valid_scores = [scores[0]['valid_score']]
            for _k in scores[1:]:
                for i, _d in enumerate(_k['details']):
                    # print(_d)
                    final_details[i]['pred'].extend(_d['pred'])
                    final_details[i]['score'].extend(_d['score'])
                final_scores.append(_k['score'])
                final_valid_scores.append(_k['valid_score'])
            avg_score = []
            for _d in final_details:
                if True in _d['score']:
                    avg_score.append(1)
                else:
                    avg_score.append(0)
            max_score = sum(avg_score) / len(avg_score) * 100
            return {
                'score': max_score,
                'all_score': final_scores,
                'valid_score': final_valid_scores,
                'details': final_details,
            }


@ICL_EVALUATORS.register_module()
class NCExactMatchEvaluator(BaseEvaluator):
    """Exact match evaluator for name conversion."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        predictions = [
            extract_chemical_data(prediction) for prediction in predictions
        ]
        references = [
            extract_chemical_data(reference) for reference in references
        ]

        cnt = 0
        valid_cnt = 0
        details = []
        for pred, ans in zip(predictions, references):
            ans = ans[-1]
            if pred:
                pred = pred[-1]
                valid_cnt += 1
            detail = {'pred': pred, 'answer': ans}
            if pred and pred.strip() == ans.strip():
                cnt += 1
                detail['correct'] = True
            else:
                detail['correct'] = False
            details.append(detail)

        score = cnt / len(predictions) * 100
        valid_score = valid_cnt / len(predictions) * 100

        return {'score': score, 'valid_score': valid_score, 'details': details}


def extract_number(text):
    pattern = re.compile(
        r'(?:<NUMBER>\s*|\\boxed\{)\s*(-?\d*\.?\d+)\s*(?:</NUMBER>|\})')
    matches = pattern.findall(text)
    return [float(match) for match in matches]


@ICL_EVALUATORS.register_module()
class RMSEEvaluator(BaseEvaluator):
    """Exact match evaluator for name conversion."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        avg_score = 0
        details = []
        for prediction, reference in zip(predictions, references):
            pred = extract_number(prediction)
            ans = extract_number(reference)
            if not pred:
                pred = 0
            else:
                pred = pred[0]
            try:
                ans = ans[0]
            except:
                raise ValueError(f'ans: {reference}')
            detail = {'pred': pred, 'answer': ans}
            rmse_score = np.sqrt(np.mean((np.array(pred) - np.array(ans))**2))
            detail['score'] = rmse_score
            avg_score += rmse_score
            details.append(detail)

        score = avg_score / len(predictions)

        return {'score': score, 'details': details}


@ICL_EVALUATORS.register_module()
class FTSEvaluator(BaseEvaluator):
    """Exact match evaluator for name conversion."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        predictions = [
            extract_chemical_data(prediction) for prediction in predictions
        ]
        references = [
            extract_chemical_data(reference) for reference in references
        ]

        avg_score = 0
        valid_cnt = 0
        details = []
        for pred, ans in zip(predictions, references):
            ans = ans[-1]
            if not pred:
                detail = {'pred': '', 'answer': ans, 'score': 0}
                details.append(detail)
                continue
            pred = pred[-1]
            detail = {'pred': pred, 'answer': ans}
            # 将 SMILES 转换为 RDKit 分子对象
            from rdkit import Chem
            mol1 = Chem.MolFromSmiles(pred)
            mol2 = Chem.MolFromSmiles(ans)
            if mol1 is None or mol2 is None:
                detail['score'] = 0
                details.append(detail)
                continue
            valid_cnt += 1
            # 生成 Morgan 指纹（等同于 ECFP4）
            # fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
            # fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
            from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
            generator = GetMorganGenerator(radius=2, fpSize=2048)
            fp1 = generator.GetFingerprint(mol1)
            fp2 = generator.GetFingerprint(mol2)
            from rdkit.Chem import DataStructs
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2) * 100
            detail['score'] = similarity
            avg_score += similarity
            details.append(detail)

        score = avg_score / len(predictions)
        valid_score = valid_cnt / len(predictions) * 100

        return {'score': score, 'valid_score': valid_score, 'details': details}


@ICL_EVALUATORS.register_module()
class MeteorEvaluator(BaseEvaluator):
    """Exact match evaluator for name conversion."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        avg_score = 0
        details = []
        for pred, ans in zip(predictions, references):
            try:
                score = (meteor_score([ans.split()], pred.split())
                         if ans and pred else 0.0)
            except AttributeError:
                self.logger = get_logger()
                self.logger.warning(f'Failed to compute METEOR'
                                    f"score:\npred='{pred}'\nans='{ans}'")
                score = 0.0
            avg_score += score
            detail = {'pred': pred, 'answer': ans, 'score': score}
            details.append(detail)

        score = avg_score / len(predictions)

        return {'score': score, 'details': details}


@TEXT_POSTPROCESSORS.register_module('smolinstruct-acc')
def smolinstruct_acc_postprocess(text: str) -> str:
    if 'yes' in text.lower():
        return '<BOOLEAN> Yes </BOOLEAN>'
    elif 'no' in text.lower():
        return '<BOOLEAN> No </BOOLEAN>'


@TEXT_POSTPROCESSORS.register_module('smolinstruct-acc-0shot')
def smolinstruct_acc_0shot_postprocess(text: str) -> str:
    # Remove <think> tags if they exist
    if '</think>' in text:
        text = text.split('</think>')[-1].strip()

    # Check for exact "yes" or "no" responses
    if text.strip().lower() == 'yes':
        return '<BOOLEAN> Yes </BOOLEAN>'
    elif text.strip().lower() == 'no':
        return '<BOOLEAN> No </BOOLEAN>'

    # Define regex patterns to match various formats of "yes" or "no"
    patterns = [
        r'\\boxed\{\s*(yes|no)\s*\}',
        r'[Th]he\s+answer\s+is\s*[\.:\'"“‘’\-]*\s*(yes|no)[\s\.,!?:;\'"”’\-]*',
        r'[Aa]nswer:\s*(yes|no)\b', r'\*\*[Aa]nswer:\*\*\s*(yes|no)\b',
        r'\*\*[Aa]nswer\*\*:\s*(yes|no)\b',
        r'<BOOLEAN>\s*(yes|no)\s*</BOOLEAN>', r'^\s*(yes|no)[\.\?!]?\s*$'
    ]
    for pattern in patterns:
        text = text.strip()
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            answer = match.group(1)  # modified
            if answer.lower() == 'yes':
                return '<BOOLEAN> Yes </BOOLEAN>'
            elif answer.lower() == 'no':
                return '<BOOLEAN> No </BOOLEAN>'

    # If no patterns matched, check for simple "yes" or "no"
    text = text.strip().lower()
    if text.startswith('yes') or text.endswith('yes'):
        return '<BOOLEAN> Yes </BOOLEAN>'
    elif text.startswith('no') or text.endswith('no'):
        return '<BOOLEAN> No </BOOLEAN>'

    # If no patterns matched, return an empty string
    return ''
