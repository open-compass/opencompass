import ast
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (matthews_corrcoef, mean_absolute_error,
                             precision_score, recall_score, roc_auc_score)

from opencompass.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class BiodataDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        final_path = os.path.join(path, f'{name}.jsonl')
        with open(final_path, 'r') as f:
            data = [json.loads(line) for line in f]
        if '-dict' in name:
            new_data = []
            for ins in data:
                new_ins = ins.copy()
                new_ins['prompt'] = (
                    ins['prompt'] +
                    'Please put your final answer with \\boxed{}' +
                    ' in json format, such as {')
                gold_keys = list(ins['ground_truth'].keys())
                for key in gold_keys:
                    new_ins['prompt'] += f"\"{key}\": xx, "
                new_ins['prompt'] = new_ins['prompt'][:-2] + '}'
                new_data.append(new_ins)
        else:
            new_data = data
        dataset = Dataset.from_list(new_data)
        return dataset


def extract_boxed_text(text):
    # 提取boxed中的内容 - 修正正则表达式以正确匹配嵌套结构
    pattern = re.compile(r'\\boxed\{((?:[^{}]|{[^{}]*})*)\}', re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None

    # 取第一个匹配的内容
    boxed_content = matches[-1].strip()

    # 只有当存在完整的\text{...}时才去掉包装，否则保持原样
    # 使用更严格的正则表达式，确保\text{...}是完整的
    clean_content = re.sub(r'\\text\{([^}]*)\}', r'\1', boxed_content)

    # 去掉LaTeX转义符
    # 处理常见的LaTeX转义字符
    clean_content = re.sub(r'\\(.)', r'\1', clean_content)

    return clean_content


@ICL_EVALUATORS.register_module()
class BiodataClsEvaluator(BaseEvaluator):
    """F1 score for fasta."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        ans_dict = {
            'positive': 'yes',
            'negative': 'no',
        }

        correct = []
        details = []
        for pred, ans in zip(predictions, references):
            pred = extract_boxed_text(pred)
            if not pred:
                detail = {'pred': pred, 'answer': ans}
                detail['score'] = 0
                details.append(detail)
                correct.append(detail['score'])
                continue
            else:
                pred = pred.lower()
            if ans in ans_dict:
                ans = ans_dict[ans]
            if pred in ans_dict:
                pred = ans_dict[pred]
            detail = {'pred': pred, 'answer': ans}
            detail['score'] = 100 if ans in pred else 0
            details.append(detail)
            correct.append(detail['score'])

        score = sum(correct) / len(correct) if correct else 0.0

        return {'score': score, 'details': details}


def extract_number(text):
    pattern = re.compile(
        r'(?:<NUMBER>\s*|\\boxed\{)\s*(-?\d*\.?\d+)\s*(?:</NUMBER>|\})')
    matches = pattern.findall(text)
    if not matches:
        return None
    return [float(match) for match in matches][-1]


@ICL_EVALUATORS.register_module()
class BiodataRMSEEvaluator(BaseEvaluator):
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
            ans = reference
            detail = {'pred': pred, 'answer': ans}
            if not pred:
                pred = 0
            rmse_score = np.sqrt(np.mean((np.array(pred) - np.array(ans))**2))
            detail['score'] = rmse_score
            avg_score += rmse_score
            details.append(detail)

        score = avg_score / len(predictions)

        return {'score': score, 'details': details}


def extract_dict_text(text):
    pattern = re.compile(r'\{[^{}]*\}', re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None
    return [match for match in matches][-1]


@ICL_EVALUATORS.register_module()
class BiodataDictEvaluator(BaseEvaluator):
    """F1 score for fasta."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        correct = []
        details = []
        for pred, ans in zip(predictions, references):
            if '</think>' in pred:
                pred = pred.split('</think>')[-1]
            pred = extract_dict_text(pred)
            if pred:
                try:
                    pred = json.loads(pred)
                except Exception:
                    try:
                        pred = ast.literal_eval(pred)
                    except Exception:
                        pred = None
            detail = {'pred': pred, 'answer': ans}
            if not pred or not isinstance(pred,
                                          dict) or pred.keys() != ans.keys():
                detail['score'] = 10
                details.append(detail)
                correct.append(detail['score'])
                continue
            cur_score = []
            for key in pred.keys():
                try:
                    pred_num = float(pred[key])
                except Exception:
                    pred_num = 0
                ans_num = float(ans[key])
                rmse_score = np.sqrt(
                    np.mean((np.array(pred_num) - np.array(ans_num))**2))
                cur_score.append(rmse_score)
            detail['score'] = sum(cur_score) / len(cur_score)
            details.append(detail)
            correct.append(detail['score'])

        score = sum(correct) / len(correct) if correct else 0.0

        return {'score': score, 'details': details}


@ICL_EVALUATORS.register_module()
class BiodataStringEvaluator(BaseEvaluator):
    """F1 score for fasta."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        all_f1 = []
        details = []
        for pred, ans in zip(predictions, references):
            if '</think>' in pred:
                pred = pred.split('</think>')[-1]
            pred = extract_boxed_text(pred)
            detail = {'pred': pred, 'answer': ans}
            if not pred:
                detail['f1'] = 0
                all_f1.append(0)
                continue
            pred_set = set(p.lower().strip() for p in pred.split(',')
                           if p.strip())
            ans_set = set(a.lower().strip() for a in ans.split(',')
                          if a.strip())

            # 计算交集、并集
            intersection = pred_set & ans_set
            # 计算精确率、召回率
            precision = len(intersection) / len(pred_set) if pred_set else 0
            recall = len(intersection) / len(ans_set) if ans_set else 0
            # 计算F1 score
            f1 = 2 * precision * recall / (precision + recall) if (
                precision + recall) > 0 else 0

            detail['f1'] = f1 * 100
            details.append(detail)
            all_f1.append(detail['f1'])

        final_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
        return {'f1': final_f1, 'details': details}


def dedup_ec_codes(ec_numer_list):
    """
    删除易被更泛化 EC 号覆盖的、更具体的 EC 号。

    规则示例：
        EC3.5.4.9  与 EC3.5.4.- 同时出现 → 去掉 EC3.5.4.9
        EC3.5.4.- 与 EC3.5.-.- 同时出现 → 去掉 EC3.5.4.-

    参数
    ----
    codes : List[str]
        原始 EC 号列表，元素格式须满足 ECa.b.c.d，其中 a–d 可以是数字或 '-'

    返回
    ----
    List[str]
        去重后的 EC 号列表，保持原有顺序
    """
    EC_PATTERN = re.compile(r'^ec(\d+|-)\.(\d+|-)\.(\d+|-)\.(\d+|-)$')
    # 先做一次规范化，保留顺序
    normalized = [c.strip() for c in ec_numer_list]
    remaining = set(normalized)  # 用集合便于快速查询

    for code in normalized:
        if code not in remaining:  # 可能在之前的循环里被删掉
            continue

        m = EC_PATTERN.match(code)
        if not m:
            # 不是合法 EC 格式，保留原状
            continue

        parts = list(
            m.groups())  # ['3', '5', '4', '9']  或 ['3', '5', '4', '-']
        # 依次生成更泛化的版本：EC3.5.4.-, EC3.5.-.-, EC3.-.-.-（不含自身）
        for i in range(3, 0, -1):
            generalized = parts[:i] + ['-'] * (4 - i)
            gen_code = 'ec' + '.'.join(generalized)
            if gen_code in remaining and gen_code != code:
                # 如果集合里已存在更泛化的版本，删除当前更具体的
                remaining.discard(code)
                break

    # 按原顺序返回仍保留的条目
    return [c for c in normalized if c in remaining]


def count_f1_max(pred, target):
    """
    F1 score with the optimal threshold.
    Handles cases where either predictions or targets are empty.

    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`

    Returns:
        float: The maximum F1 score or 0.0 if inputs are empty.
    """
    # Check if either pred or target is empty
    if pred.numel() == 0 or target.numel() == 0:
        print('Empty input provided. Returning F1 score of 0.0.')
        return 0.0

    # Proceed with the original logic if inputs are not empty
    order = pred.argsort(descending=True, dim=1, stable=True)
    # print(f"order: {order}")
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)

    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)
    # print("isstart {}".format(is_start))
    all_order = pred.flatten().argsort(descending=True, stable=True)
    order = order + torch.arange(
        order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]

    precision = precision.flatten()
    recall = recall.flatten()

    all_precision = precision[all_order] - \
        torch.where(is_start, torch.zeros_like(precision),
                    precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - \
        torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall +
                                               1e-10)

    if torch.isnan(all_f1).any():
        print(f'NaN encountered in F1 score computation. all_f1: {all_f1}')
        return 0.0

    return all_f1.max()


@ICL_EVALUATORS.register_module()
class BiodataECNumberEvaluator(BaseEvaluator):
    """F1 score for fasta."""

    def __init__(self) -> None:
        super().__init__()
        self.ec_labels = [
            '1.4.3.-', '4.2.1.1', '2.7.6.-', '3.5.1.88', '5.4.99.-',
            '3.1.21.4', '2.3.1.48', '7.2.1.-', '1.16.3.1', '3.4.19.12',
            '1.3.8.-', '2.7.7.19', '2.4.2.-', '1.1.1.169', '2.4.2.10',
            '3.5.3.1', '3.3.2.-', '3.1.26.4', '7.1.1.9', '3.4.13.9',
            '1.1.1.100', '5.1.1.-', '3.1.22.-', '3.4.21.-', '1.11.1.-',
            '3.4.22.28', '4.6.1.18', '3.4.21.4', '6.1.1.4', '1.15.1.1',
            '3.4.19.-', '3.4.11.1', '3.4.25.1', '1.3.1.9', '2.4.99.-',
            '1.1.99.-', '1.97.1.12', '2.7.4.6', '1.17.1.-', '1.7.2.-',
            '2.7.1.71', '4.2.1.11', '3.4.21.90', '4.1.99.-', '2.8.1.1',
            '3.6.4.13', '6.3.4.4', '2.7.11.1', '2.7.7.23', '1.9.3.-',
            '2.3.1.39', '4.1.1.11', '3.2.1.26', '3.1.13.-', '1.8.99.-',
            '1.11.1.6', '4.2.1.2', '5.3.4.1', '3.2.1.4', '3.4.21.5',
            '1.14.16.-', '6.3.4.2', '2.1.1.37', '1.12.99.-', '2.1.1.361',
            '2.1.1.-', '3.1.8.1', '2.7.11.-', '2.3.2.24', '2.7.7.48',
            '3.5.1.98', '3.1.1.31', '2.7.11.22', '1.18.1.2', '3.1.13.4',
            '4.1.1.23', '3.2.2.27', '2.5.1.7', '1.14.11.-', '3.5.1.2',
            '6.3.1.2', '4.3.2.-', '4.1.2.25', '5.3.2.-', '2.7.1.1', '3.1.11.-',
            '2.7.7.4', '3.6.3.14', '4.2.1.20', '2.3.1.41', '1.18.6.1',
            '2.3.1.74', '6.3.2.19', '3.2.2.22', '2.8.4.1', '2.4.2.17',
            '2.1.1.56', '1.10.3.-', '2.5.1.54', '1.2.1.11', '4.2.1.8',
            '3.1.3.16', '1.12.7.2', '2.7.1.-', '1.17.4.1', '2.7.10.-',
            '3.1.2.14', '5.3.1.6', '3.4.21.91', '2.2.1.-', '2.7.1.2',
            '2.7.3.-', '3.1.22.4', '2.3.1.16', '4.99.1.-', '2.7.1.35',
            '3.4.22.69', '3.1.27.-', '1.12.7.-', '5.1.3.-', '2.7.7.65',
            '1.17.4.-', '5.3.1.24', '4.2.1.59', '2.5.1.10', '1.8.4.11',
            '3.4.25.-', '2.7.10.2', '5.2.1.-', '6.1.1.3', '2.7.7.49',
            '2.8.4.-', '4.1.3.-', '2.3.1.31', '1.1.1.205', '3.1.30.-',
            '3.4.23.-', '6.5.1.2', '6.1.1.15', '3.6.4.-', '6.2.1.-', '4.1.3.3',
            '2.7.7.60', '6.3.2.6', '5.1.1.3', '2.4.2.9', '1.14.13.-',
            '1.1.2.-', '1.1.1.1', '5.1.99.-', '2.8.2.-', '2.5.1.47',
            '2.7.11.24', '3.4.22.15', '2.6.1.42', '2.1.3.2', '3.2.2.9',
            '4.2.3.3', '2.6.1.-', '1.5.1.-', '2.7.7.24', '2.1.1.57',
            '6.1.1.20', '5.3.1.5', '2.7.1.25', '2.2.1.1', '3.6.1.1',
            '2.3.3.16', '6.3.4.-', '2.7.7.9', '1.18.1.-', '4.2.99.-',
            '4.1.2.4', '3.1.3.1', '5.3.3.8', '3.2.1.1', '2.7.10.1',
            '4.2.1.113', '4.2.2.2', '6.1.1.1', '3.1.3.-', '1.2.4.-',
            '1.6.99.-', '2.5.1.18', '3.4.22.29', '3.1.3.2', '1.1.1.27',
            '2.3.1.286', '1.14.15.-', '2.7.7.3', '3.1.13.2', '2.7.7.6',
            '5.4.99.18', '4.1.1.39', '2.7.4.8', '5.6.2.1', '2.3.1.-',
            '1.7.1.-', '1.6.5.2', '1.18.6.-', '4.6.1.2', '2.6.1.52', '3.1.6.-',
            '1.6.5.-', '2.3.2.31', '3.6.5.5', '2.8.3.-', '2.3.2.-', '3.2.1.18',
            '3.5.99.-', '3.1.4.35', '3.1.8.-', '2.3.1.12', '1.6.2.-',
            '2.1.1.72', '2.3.3.-', '3.4.21.92', '1.14.11.27', '2.7.11.17',
            '2.1.1.359', '2.7.13.-', '2.5.1.15', '6.3.3.-', '3.2.1.22',
            '6.1.1.6', '4.3.3.7', '3.4.11.18', '4.2.3.4', '3.1.2.-', '2.4.2.8',
            '4.1.1.48', '3.7.1.-', '3.1.4.53', '7.2.2.-', '2.7.6.5', '3.6.5.3',
            '4.3.3.-', '2.7.1.21', '3.1.4.-', '1.11.1.24', '3.6.4.10',
            '1.14.14.1', '3.5.1.60', '3.2.1.52', '1.16.3.-', '3.1.26.3',
            '3.4.24.69', '3.5.1.11', '2.1.1.193', '1.7.2.1', '1.14.99.-',
            '3.6.5.2', '2.7.7.n1', '4.1.1.-', '2.3.2.26', '2.4.1.15',
            '2.5.1.-', '3.1.3.11', '1.14.11.67', '2.3.1.180', '2.4.2.7',
            '3.6.4.12', '2.5.1.19', '1.1.1.-', '1.8.1.9', '1.9.3.1',
            '2.7.1.15', '2.7.1.11', '3.4.11.-', '2.1.2.-', '7.6.2.-',
            '3.5.1.28', '3.8.1.5', '6.3.4.13', '1.11.1.9', '1.13.11.-',
            '4.1.2.13', '3.4.16.4', '6.1.1.7', '1.1.3.-', '2.4.1.129',
            '3.1.1.29', '3.4.21.53', '3.6.5.-', '3.1.1.4', '1.1.1.86',
            '3.2.2.-', '2.6.1.1', '4.2.99.18', '5.5.1.-', '2.7.11.30',
            '3.1.26.-', '1.8.1.4', '1.14.14.-', '3.2.1.14', '3.5.4.-',
            '2.1.2.1', '3.2.1.3', '1.97.1.-', '6.3.4.14', '2.7.12.1',
            '2.7.11.25', '2.5.1.17', '6.3.5.2', '6.3.2.1', '3.2.1.78',
            '3.1.11.2', '1.6.99.3', '2.1.2.2', '1.1.1.42', '7.1.1.8',
            '7.1.1.2', '3.1.2.2', '3.4.22.46', '3.1.3.36', '2.4.1.-',
            '3.1.3.33', '4.3.2.2', '1.14.12.-', '1.13.12.-', '3.4.22.-',
            '5.3.1.1', '4.2.1.-', '3.2.1.169', '3.2.1.17', '6.5.1.1',
            '1.1.1.35', '1.3.5.-', '1.2.1.3', '2.5.1.1', '7.2.2.8', '6.3.1.-',
            '2.5.1.78', '2.7.7.50', '2.1.3.-', '2.7.7.-', '5.4.3.8', '2.7.2.4',
            '1.10.3.2', '3.1.1.3', '3.6.1.15', '5.6.2.2', '3.1.3.3',
            '3.2.1.20', '3.6.1.9', '2.3.2.27', '3.6.1.23', '6.1.1.-',
            '4.2.1.10', '3.4.13.-', '6.3.2.4', '1.1.1.2', '3.4.21.107',
            '1.6.99.1', '2.7.4.9', '1.15.1.-', '3.6.1.34', '1.3.1.-',
            '3.6.1.55', '3.4.24.-', '3.6.1.-', '4.1.1.50', '4.2.2.-',
            '3.3.1.1', '3.4.22.1', '3.1.4.11', '3.5.1.1', '3.3.1.-',
            '1.1.1.267', '3.2.1.55', '1.1.1.25', '3.6.1.7', '2.7.13.3',
            '2.7.1.40', '2.3.1.9', '1.7.3.-', '5.4.2.-', '1.7.1.17', '3.2.2.6',
            '4.1.1.33', '1.8.5.-', '5.3.3.-', '3.2.1.31', '6.3.5.-',
            '1.14.19.-', '6.1.1.11', '1.12.99.6', '1.4.1.-', '4.6.1.1',
            '3.1.3.86', '3.2.1.91', '4.3.2.10', '3.4.16.-', '3.1.3.5',
            '3.5.4.4', '6.4.1.-', '1.17.1.8', '2.5.1.16', '4.3.1.-',
            '3.4.23.16', '6.3.3.1', '3.2.1.73', '5.1.3.13', '1.2.1.12',
            '1.6.2.4', '6.1.1.17', '2.3.1.1', '3.5.3.-', '3.2.1.8', '2.1.1.45',
            '3.2.1.21', '5.1.3.2', '2.3.1.129', '2.7.2.-', '5.3.1.-',
            '2.7.2.8', '2.4.2.1', '1.14.14.18', '3.5.1.5', '3.5.4.38',
            '5.4.3.-', '6.5.1.-', '2.7.12.2', '2.5.1.55', '1.8.1.7',
            '3.1.21.-', '1.8.4.12', '1.11.1.15', '1.1.1.85', '3.6.1.3',
            '2.7.1.33', '2.7.8.7', '3.1.3.25', '3.2.1.96', '7.1.1.-',
            '3.2.1.39', '2.4.2.3', '3.5.4.9', '2.2.1.2', '3.6.3.-', '3.5.1.-',
            '1.11.1.7', '1.5.1.5', '2.4.2.19', '1.8.3.2', '1.3.99.-',
            '1.5.3.-', '5.6.1.-', '1.8.1.-', '2.7.3.9', '2.5.1.6', '3.4.21.62',
            '1.8.4.-', '5.3.1.16', '2.3.2.2', '3.4.17.-', '2.7.4.-', '3.2.1.-',
            '6.3.1.5', '4.3.3.6', '2.1.3.3', '1.5.1.3', '3.5.2.3', '5.4.2.11',
            '2.7.7.8', '2.8.1.7', '2.7.7.7', '3.2.1.37', '2.7.12.-', '5.3.1.9',
            '1.2.7.-', '6.1.1.2', '7.1.2.-', '2.3.1.5', '3.4.14.-', '6.1.1.10',
            '1.16.1.-', '2.1.1.228', '3.5.2.6', '2.1.1.354', '2.7.4.3',
            '4.1.2.-', '4.4.1.5', '5.3.4.-', '4.6.1.-', '5.1.1.1', '3.4.24.3',
            '4.4.1.-', '1.3.7.-', '2.3.1.117', '5.4.99.5', '1.2.1.-',
            '2.4.2.30', '1.14.18.-', '3.1.1.1', '3.1.3.48', '3.4.21.98',
            '5.6.2.-', '3.1.26.5', '7.2.1.1', '2.7.11.12', '1.3.3.-',
            '2.7.7.18', '3.1.1.-', '5.2.1.8', '2.7.1.69', '1.1.1.37',
            '3.6.4.6', '3.1.4.17', '2.7.11.13', '3.5.2.-', '4.2.1.17',
            '2.7.2.3', '4.2.3.-', '5.3.99.-', '3.1.1.72', '2.3.1.179',
            '3.2.1.23', '1.14.13.25', '3.1.26.13', '3.8.1.-', '4.1.1.20',
            '2.7.11.26', '6.1.1.21', '2.7.11.21', '2.7.4.22', '1.8.3.-',
            '2.3.1.57', '1.3.5.1', '3.1.1.53', '7.1.2.2', '6.4.1.2', '2.7.8.-',
            '6.3.2.-', '2.8.1.-', '3.5.4.5', '4.6.1.12', '2.3.2.23'
        ]

    def ec_to_multihot(self, ec_list, ec_labels):
        multihot = torch.zeros(len(ec_labels))
        if not ec_list:  # Check if ec_list is empty
            return multihot
        multihot = torch.zeros(len(ec_labels))
        for ec in ec_list:
            if ec in ec_labels:
                idx = ec_labels.index(ec)
                multihot[idx] = 1
        return multihot

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        all_preds = []
        all_labels = []
        details = []
        for pred, ans in zip(predictions, references):
            if '</think>' in pred:
                pred = pred.split('</think>')[-1]
            pred = extract_boxed_text(pred)
            detail = {'pred': pred, 'answer': ans}
            if not pred:
                result_ec = []
            else:
                result_ec = re.findall(r'\d+\.\d+\.\d+\.\-?\d*', pred)
            label_ec = re.findall(r'\d+\.\d+\.\d+\.\-?\d*', ans)

            # Convert EC numbers to multi-hot vectors
            pred_multihot = self.ec_to_multihot(result_ec, self.ec_labels)
            label_multihot = self.ec_to_multihot(label_ec, self.ec_labels)
            # Store the results
            all_preds.append(pred_multihot)
            all_labels.append(label_multihot)
            cur_fmax_score = count_f1_max(torch.stack([pred_multihot]),
                                          torch.stack([label_multihot]))
            detail['score'] = cur_fmax_score.item()
            details.append(detail)

        # Stack the predictions and targets for batch processing
        all_preds = torch.stack(all_preds)
        all_labels = torch.stack(all_labels)

        # Compute the Fmax score
        try:
            fmax_score = count_f1_max(all_preds, all_labels)
        except ValueError:
            fmax_score = None

        return {'score': fmax_score.item() * 100, 'details': details}


@LOAD_DATASET.register_module()
class BiodataTaskDataset(BaseDataset):

    @staticmethod
    def load(path: str, task: str):
        hint_dict = {
            'cls':
            'Please use "yes" or "no" as your answer, '
            'and put your answer within \\boxed{}.',
            'number':
            'Please put your answer number within \\boxed{}.',
            'Function EC':
            'Please put the final enzyme within \\boxed{} '
            'using "EC number", such as "ECx.x.x.x".'
            ' Please split by "," if there are multiple enzymes.',
            'Non-coding RNA Function Classification':
            'You should choose from '
            '[leader, tRNA, 5_8S_rRNA, ribozyme, CD-box, Intron_gpII, IRES,'
            ' 5S_rRNA, scaRNA, miRNA, riboswitch, Intron_gpI, HACA-box].'
            ' Please put your final answer within \\boxed{}.',
            'Modification Prediction':
            'You should choose from '
            '[AtoI, m6A, none, m1A, m5C, m5U, m6Am, m7G, Cm, Am, Gm, Um, Psi].'
            ' Please put your final answer within \\boxed{},'
            ' split by "," if there are multiple answers.',
        }

        path = get_data_path(path)
        path = os.path.join(path, f'{task}.jsonl')

        with open(path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        new_data = []
        for ins in data:
            ans_type = ins['meta_data']['ans_type']
            new_ins = ins.copy()
            if ans_type == 'dict':
                new_ins['prompt'] = (
                    ins['prompt'] +
                    'Please put your final answer with \\boxed{}' +
                    ' in json format, such as {')
                gold_keys = list(ins['ground_truth'].keys())
                for key in gold_keys:
                    new_ins['prompt'] += f"\"{key}\": xx, "
                new_ins['prompt'] = new_ins['prompt'][:-2] + '}'
            elif ans_type in ['cls', 'number']:
                new_ins[
                    'prompt'] = new_ins['prompt'] + '\n' + hint_dict[ans_type]
            else:
                assert ans_type == 'string', f'Wrong answer type! {ans_type}'
                new_ins['prompt'] = new_ins['prompt'] + '\n' + hint_dict[
                    ins['meta_data']['task']]
            new_data.append(new_ins)

        dataset = Dataset.from_list(new_data)
        return dataset


def pearson_correlation_coefficient(y_true, y_pred):
    """
    计算皮尔逊相关系数 (PCC)
    适用于回归问题
    """
    # Convert the label and result values to numpy arrays
    result_values = np.array(y_pred).flatten()
    label_values = np.array(y_true).flatten()

    # Identify explicitly assigned infinity values
    near_infinity_mask = np.isinf(result_values)

    if near_infinity_mask.any():
        print(f'Found {sum(near_infinity_mask)} result values near infinity.'
              ' These will be assigned a Spearman score of 0.')

    # Exclude near-infinity pairs from the main calculation
    valid_mask = ~near_infinity_mask & np.isfinite(
        result_values) & np.isfinite(label_values)
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    # Compute Spearman correlation for valid values
    if len(valid_result_values) > 0:
        correlation, _ = pearsonr(valid_label_values, valid_result_values)
    else:
        correlation = 0  # Fallback if no valid pairs
        print('No valid result values. Assign the spearman to 0.')

    # Combine Spearman score for valid and infinity values
    total_data_points = len(result_values)
    total_valid_points = valid_mask.sum()
    num_infinity_values = near_infinity_mask.sum()

    if num_infinity_values > 0:
        final_score = (correlation * total_valid_points +
                       0 * num_infinity_values) / total_data_points
    else:
        final_score = correlation  # Edge case: no near-infinity values
    return final_score


def spearman_correlation_coefficient(y_true, y_pred):
    """
    计算斯皮尔曼等级相关系数
    适用于非线性单调关系
    """
    # Convert the label and result values to numpy arrays
    result_values = np.array(y_pred).flatten()
    label_values = np.array(y_true).flatten()

    # Identify explicitly assigned infinity values
    near_infinity_mask = np.isinf(result_values)

    if near_infinity_mask.any():
        print(f'Found {sum(near_infinity_mask)} result values near infinity.'
              f' These will be assigned a Spearman score of 0.')

    # Exclude near-infinity pairs from the main calculation
    valid_mask = ~near_infinity_mask & np.isfinite(
        result_values) & np.isfinite(label_values)
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    # Compute Spearman correlation for valid values
    if len(valid_result_values) > 0:
        spearman, _ = spearmanr(valid_label_values, valid_result_values)
    else:
        spearman = 0  # Fallback if no valid pairs
        print('No valid result values. Assign the spearman to 0.')

    # Combine Spearman score for valid and infinity values
    total_data_points = len(result_values)
    total_valid_points = valid_mask.sum()
    num_infinity_values = near_infinity_mask.sum()

    if num_infinity_values > 0:
        final_spearman_score = (spearman * total_valid_points +
                                0 * num_infinity_values) / total_data_points
    else:
        final_spearman_score = spearman  # Edge case: no near-infinity values
    return final_spearman_score


def r_squared(y_true, y_pred):
    # Convert the label and result values to numpy arrays
    result_values = np.array(y_pred).flatten()
    label_values = np.array(y_true).flatten()

    # Identify explicitly assigned infinity values
    near_infinity_mask = np.isinf(result_values)

    if near_infinity_mask.any():
        print(f'Found {sum(near_infinity_mask)} result values near infinity.'
              f'These will be assigned an R2 score of 0.')

    # Exclude near-infinity pairs from the main calculation
    valid_mask = ~near_infinity_mask & np.isfinite(
        result_values) & np.isfinite(label_values)
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    # Compute Pearson correlation coefficient for valid values
    if len(valid_result_values) > 0:
        try:
            pcc, _ = pearsonr(valid_label_values, valid_result_values)
            R2 = pcc**2
        except Exception as e:
            print(f'Error in computing R2: {e}. Assign the R2 to inf.')
            R2 = np.inf  # Fallback to inf if computation fails
    else:
        R2 = 0  # Fallback if no valid pairs
        print('No valid result values. Assign the R2 to 0.')

    # Combine R2 score for valid and infinity values
    total_data_points = len(result_values)
    total_valid_points = valid_mask.sum()
    num_infinity_values = near_infinity_mask.sum()

    if num_infinity_values > 0:
        final_R2_score = (R2 * total_valid_points +
                          0 * num_infinity_values) / total_data_points
    else:
        final_R2_score = R2  # Edge case: no near-infinity values
    return final_R2_score


def multiple_label_auc(y_true, y_pred):
    """
    y_true: (N, C) 二值多标签真值矩阵
    y_pred: (N, C) 连续分数/概率，数值越大越倾向为该标签
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_pred)
    assert y_true.shape == y_score.shape,\
        (f'Different shape y_true.shape={y_true.shape},'
         f'y_score.shape={y_score.shape}')
    assert y_true.ndim == 2,\
        f'y_true.ndim is {y_true.ndim}, excepted 2'
    N, C = y_true.shape

    per_class = []
    for c in range(C):
        yt = y_true[:, c]
        ys = y_score[:, c]
        # 若该类在所有样本里全 0 或全 1，则 AUC 不可计算，跳过
        if yt.min() == yt.max():
            per_class.append(np.nan)
            continue
        per_class.append(roc_auc_score(yt, ys))
    per_class = np.array(per_class, dtype=float)
    macro_auc = np.nanmean(per_class)  # 仅对有效类取均值
    return float(macro_auc)


def mixed_score(y_true, y_pred, low_range=(30, 1e3)):
    """
    计算Mixed Score，结合MAE、Range-MAE和F1 Score

    Args:
        y_true: 真实值数组
        y_pred: 预测值数组
        low_range: 低范围区间，默认为(0, 30)

    Returns:
        mixed_score: 最终的Mixed Score
        components: 包含各个组件分数的字典
    """
    # Convert the label and result values to numeric arrays
    # using pandas to handle non-numeric entries
    result_values = pd.to_numeric(y_pred, errors='coerce').flatten()
    label_values = pd.to_numeric(y_true, errors='coerce').flatten()

    # Identify near-infinity values
    near_infinity_mask = np.abs(result_values) > low_range[1]
    if near_infinity_mask.any():
        print(
            f'Warning: Found {sum(near_infinity_mask)} result values too large'
            f'will be assigned a mixed score of 0.')
        print(f'Large result values: {result_values[near_infinity_mask]} ')
        print(
            f'Warning: Found {sum(near_infinity_mask)} result values too large'
            'will be assigned a mixed score of 0.'
            f'Large result values: {result_values[near_infinity_mask]} ')

    # Exclude near-infinity pairs from the main calculation
    valid_mask = ~near_infinity_mask & np.isfinite(
        result_values) & np.isfinite(label_values)
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    # Assign a mixed score of 0 to near-infinity pairs
    num_infinity_values = near_infinity_mask.sum()
    if num_infinity_values > 0:
        mixed_score_infinity = 0

    # Convert to binary based on the threshold for valid values
    label_binary = (valid_label_values < low_range[0]).astype(int)
    result_binary = (valid_result_values < low_range[0]).astype(int)

    # Compute precision, recall, F1 score for valid values
    precision = precision_score(label_binary, result_binary, average='binary')
    recall = recall_score(label_binary, result_binary, average='binary')
    f1 = 2 * precision * recall / (precision + recall) if (precision +
                                                           recall) != 0 else 0
    print('F1:', f1)

    try:
        # Compute mean absolute error (MAE) for valid values
        mae = mean_absolute_error(valid_label_values, valid_result_values)
        print('MAE:', mae)
    except ValueError as e:
        print(f'Error in computing MAE: {e}')
        mae = np.inf  # Fallback to infinity if error occurs

    # Mask to keep only values in the range [0, threshold] for valid values
    mask = (valid_result_values >= 0) & (valid_result_values <= low_range[0])
    if mask.sum() > 0:
        range_mae = mean_absolute_error(valid_label_values[mask],
                                        valid_result_values[mask])
    else:
        range_mae = 100  # Fallback if no values within the range
    print('Range MAE:', range_mae)

    # Ensure MAE and range_mae are within reasonable bounds to avoid overflow
    mae = min(mae, 100)
    range_mae = min(range_mae, 100)

    # Compute mixed score for valid values
    mixed_score_valid = (1 - mae / 100) * 0.5 + (1 -
                                                 range_mae / 100) * f1 * 0.5
    print(
        f'(1 - mae / 100) * 0.5={(1 - mae / 100) * 0.5}\n'
        f'(1 - range_mae / 100)={(1 - range_mae / 100)}\n'
        f'(1 - range_mae / 100) * f1 * 0.5={(1 - range_mae / 100) * f1 * 0.5}')
    print(
        f'(1 - mae / 100) * 0.5={(1 - mae / 100) * 0.5}\n'
        f'(1 - range_mae / 100)={(1 - range_mae / 100)}\n'
        f'(1 - range_mae / 100) * f1 * 0.5={(1 - range_mae / 100) * f1 * 0.5}')

    # Compute the final mixed score,
    # averaging in the score for the near-infinity pairs
    total_data_points = len(result_values)
    total_valid_points = valid_mask.sum()

    if num_infinity_values > 0:
        final_mixed_score = (
            mixed_score_valid * total_valid_points +
            mixed_score_infinity * num_infinity_values) / total_data_points
    else:
        # Edge case: no near-infinity values
        final_mixed_score = mixed_score_valid

    return final_mixed_score


@ICL_EVALUATORS.register_module()
class BiodataMCCEvaluator(BaseEvaluator):
    """F1 score for fasta."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        ans_dict = {'positive': 1, 'negative': 0, 'yes': 1, 'no': 0}

        details = []
        y_pred = []
        y_true = []
        for pred, ans in zip(predictions, references):
            if '</think>' in pred:
                pred = pred.split('</think>')[-1]
            pred = extract_boxed_text(pred)
            detail = {'pred': pred, 'answer': ans}
            ans_label = ans_dict[ans.lower()]
            if not pred or pred.lower() not in ans_dict:
                pred_label = 1 - ans_label
            else:
                pred_label = ans_dict[pred.lower()]
            y_pred.append(pred_label)
            y_true.append(ans_label)
            detail['score'] = 1 if pred_label == ans_label else 0
            details.append(detail)

        score = matthews_corrcoef(y_true, y_pred)
        return {'mcc': score * 100, 'details': details}


@ICL_EVALUATORS.register_module()
class BiodataPCCEvaluator(BaseEvaluator):
    """F1 score for fasta."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        details = []
        y_true = defaultdict(list)
        y_pred = defaultdict(list)
        for pred, ans in zip(predictions, references):
            if '</think>' in pred:
                pred = pred.split('</think>')[-1]
            pred = extract_dict_text(pred)
            if pred:
                try:
                    pred = json.loads(pred)
                except Exception:
                    try:
                        pred = ast.literal_eval(pred)
                    except Exception:
                        pred = None
            detail = {'pred': pred, 'answer': ans}
            if not pred or not isinstance(pred,
                                          dict) or pred.keys() != ans.keys():
                pred = dict()
                for key in ans.keys():
                    pred[key] = np.inf
            for key in pred.keys():
                try:
                    pred_num = float(pred[key])
                except Exception:
                    pred_num = np.inf
                ans_num = float(ans[key])
                y_true[key].append(ans_num)
                y_pred[key].append(pred_num)
            details.append(detail)
        scores = []
        for key in y_true.keys():
            scores.append(
                pearson_correlation_coefficient(y_true[key], y_pred[key]))
        score = sum(scores) / len(scores)

        return {'pcc': score * 100, 'details': details}


@ICL_EVALUATORS.register_module()
class BiodataSpearmanEvaluator(BaseEvaluator):
    """Exact match evaluator for name conversion."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        details = []
        y_true = []
        y_pred = []
        for pred, ans in zip(predictions, references):
            if '</think>' in pred:
                pred = pred.split('</think>')[-1]
            pred = extract_number(pred)
            detail = {'pred': pred, 'answer': ans}
            if not pred:
                pred = np.inf
            y_true.append(ans)
            y_pred.append(pred)
            details.append(detail)

        score = spearman_correlation_coefficient(y_true, y_pred)

        return {'spearman': score * 100, 'details': details}


@ICL_EVALUATORS.register_module()
class BiodataMixedScoreEvaluator(BaseEvaluator):
    """Exact match evaluator for name conversion."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        details = []
        y_true = []
        y_pred = []
        for pred, ans in zip(predictions, references):
            if '</think>' in pred:
                pred = pred.split('</think>')[-1]
            pred = extract_number(pred)
            detail = {'pred': pred, 'answer': ans}
            if not pred:
                pred = 0
            y_true.append(ans)
            y_pred.append(pred)
            detail['score'] = mixed_score([ans], [pred]) * 100
            details.append(detail)

        score = mixed_score(y_true, y_pred)

        return {'Mixed': score * 100, 'details': details}


@ICL_EVALUATORS.register_module()
class BiodataR2Evaluator(BaseEvaluator):
    """F1 score for fasta."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        details = []
        if isinstance(references[0], dict):
            y_true = defaultdict(list)
            y_pred = defaultdict(list)
            for pred, ans in zip(predictions, references):
                if '</think>' in pred:
                    pred = pred.split('</think>')[-1]
                pred = extract_dict_text(pred)
                if pred:
                    try:
                        pred = json.loads(pred)
                    except Exception:
                        try:
                            pred = ast.literal_eval(pred)
                        except Exception:
                            pred = None
                if not pred or not isinstance(
                        pred, dict) or pred.keys() != ans.keys():
                    pred = dict()
                    for key in ans.keys():
                        pred[key] = np.inf
                else:
                    for key in pred.keys():
                        try:
                            pred[key] = float(pred[key])
                        except Exception:
                            pred[key] = np.inf
                detail = {'pred': pred, 'answer': ans}
                for key in pred.keys():
                    ans_num = float(ans[key])
                    y_true[key].append(ans_num)
                    y_pred[key].append(pred[key])
                details.append(detail)

            scores = []
            for key in y_true.keys():
                scores.append(r_squared(y_true[key], y_pred[key]))
                a = r_squared(y_true[key], y_pred[key])
                if a != a:
                    print(key)
                    print(f'y_true: {y_true[key]}')
                    print(f'y_pred: {y_pred[key]}')
            score = sum(scores) / len(scores)
        else:
            assert isinstance(references[0], float)
            y_true = []
            y_pred = []
            for pred, ans in zip(predictions, references):
                if '</think>' in pred:
                    pred = pred.split('</think>')[-1]
                pred = extract_number(pred)
                detail = {'pred': pred, 'answer': ans}
                if not pred:
                    pred = np.inf
                y_true.append(ans)
                y_pred.append(pred)
                details.append(detail)

            score = r_squared(y_true, y_pred)

        return {'r^2': score * 100, 'details': details}


@ICL_EVALUATORS.register_module()
class BiodataAucEvaluator(BaseEvaluator):
    """AUC score for biodata multi-label
    classification with dynamic label discovery."""

    def __init__(self, predefined_labels=None) -> None:
        super().__init__()
        # 可以预定义标签，也可以动态发现
        self.predefined_labels = set(
            predefined_labels) if predefined_labels else None

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        details = []
        # 用于存储所有样本的二进制标签和预测分数
        y_true = []
        y_pred = []
        for pred, ans in zip(predictions, references):
            # 处理预测结果
            if '</think>' in pred:
                pred = pred.split('</think>')[-1]
            pred = extract_boxed_text(pred)
            detail = {'pred': pred, 'answer': ans}
            if not pred:
                # 为空预测创建全零向量
                cur_pred = [0 for _ in self.predefined_labels]
            else:
                pred_set = set(p.lower().strip() for p in pred.split(',')
                               if p.strip())
                cur_pred = [
                    0 if p in pred_set else 1 for p in self.predefined_labels
                ]
            ans_set = set(a.lower().strip() for a in ans.split(',')
                          if a.strip())
            cur_true = [
                0 if p in ans_set else 1 for p in self.predefined_labels
            ]
            y_true.append(cur_true)
            y_pred.append(cur_pred)
            detail['auc'] = multiple_label_auc([cur_true], [cur_pred]) * 100
            details.append(detail)

        score = multiple_label_auc(y_true, y_pred)

        return {'auc': score * 100, 'details': details}


@ICL_EVALUATORS.register_module()
class BiodataAccEvaluator(BaseEvaluator):
    """F1 score for fasta."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        ans_dict = {
            'positive': 'yes',
            'negative': 'no',
        }

        scores = []
        details = []
        for pred, ans in zip(predictions, references):
            pred = extract_boxed_text(pred)
            if not pred:
                detail = {'pred': pred, 'answer': ans}
                detail['acc'] = 0
                details.append(detail)
                scores.append(detail['acc'])
                continue
            else:
                pred = pred.lower()
            if ans in ans_dict:
                ans = ans_dict[ans]
            if pred in ans_dict:
                pred = ans_dict[pred]
            detail = {'pred': pred, 'answer': ans}

            detail['acc'] = 100 if pred.lower() == ans.lower() else 0
            details.append(detail)
            scores.append(detail['acc'])

        score = sum(scores) / len(scores) if scores else 0.0

        return {'acc': score, 'details': details}


if __name__ == '__main__':
    dedup_ec_codes('EC1.5.1.5,EC1.5.1.-,EC3.5.4.9,EC3.5.4.-')
