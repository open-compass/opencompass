import json
import os
import re
import sys

from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

# moleculariq_core 与本文件同目录，需手动加入 sys.path
_DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
if _DATASET_DIR not in sys.path:
    sys.path.insert(0, _DATASET_DIR)

from moleculariq_core.rewards.constraint_reward import \
    multi_constraint_generation_reward  # noqa: E402
from moleculariq_core.rewards.count_reward import \
    multi_count_dict_reward  # noqa: E402
from moleculariq_core.rewards.index_reward import \
    multi_index_identification_reward  # noqa: E402


def extract_json(prediction):
    """从模型输出的 <answer>...</answer> 标签中提取内容。

    模型输出格式示例：
      <answer>"ring count": 2, "halogen indices": [3, 7]</answer>
      <answer>"smiles": "CC(O)C"</answer>

    返回值：
      - dict：内容可解析为 JSON 对象时
      - str：内容为裸数字或 SMILES 等纯字符串时
      - None：未找到标签或内容为空时
    """
    if not isinstance(prediction, str):
        return None
    match = re.search(r'<answer>(.*?)</answer>', prediction,
                      re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    content = match.group(1).strip()
    if not content:
        return None
    # 已是完整 JSON 对象
    if content.startswith('{'):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
    # 裸 key-value 对，补上大括号
    try:
        return json.loads('{' + content + '}')
    except json.JSONDecodeError:
        pass
    # 模型有时输出 "key": value} (有结尾 } 但无开头 {)，去掉多余 } 再试
    if not content.startswith('{') and content.endswith('}'):
        try:
            return json.loads('{' + content[:-1] + '}')
        except json.JSONDecodeError:
            pass
    # 兜底：返回原始字符串（裸数字、SMILES 等）
    return content


@LOAD_DATASET.register_module()
class MoleculariqDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, **kwargs):
        new_data = []
        filename = name if name.endswith('.jsonl') else name + '.jsonl'
        filepath = os.path.join(get_data_path(path), filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # generation 任务 target 为 null，ground_truth 取 constraints
                ground_truth = item.get('target') or item.get('constraints')
                new_data.append({
                    'id': item['uid'],
                    'prompt': item['question'],
                    'ground_truth': ground_truth,
                })
        return Dataset.from_list(new_data)


@ICL_EVALUATORS.register_module()
class MoleculariqCountEvaluator(BaseEvaluator):
    """Count 任务评测：所有属性计数完全正确才得 1 分。

    输出指标：
      score  – 精确匹配准确率 (%)
      valid  – 可解析为 JSON 的预测比例 (%)
    """

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        scores, valids, details = [], [], []
        for pred_raw, ref in zip(predictions, references):
            pred = extract_json(pred_raw)
            if pred is None:
                scores.append(0.0)
                valids.append(0)
                details.append({
                    'pred': None,
                    'answer': ref,
                    'score': 0,
                    'correct': 0
                })
                continue

            valids.append(1)
            reward = multi_count_dict_reward(pred, ref)
            scores.append(reward)
            details.append({
                'pred': pred,
                'answer': ref,
                'score': reward,
                'correct': int(reward == 1.0)
            })

        n = len(scores)
        return {
            'score': sum(scores) / n * 100 if n > 0 else 0.0,
            'valid': sum(valids) / n * 100 if n > 0 else 0.0,
            'details': details,
        }


@ICL_EVALUATORS.register_module()
class MoleculariqIndexEvaluator(BaseEvaluator):
    """Index 任务评测：所有属性的原子索引集合完全正确才得 1 分。

    输出指标：
      score  – 精确匹配准确率 (%)
      valid  – 可解析为 JSON 的预测比例 (%)
    """

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        scores, valids, details = [], [], []
        for pred_raw, ref in zip(predictions, references):
            pred = extract_json(pred_raw)
            if pred is None:
                scores.append(0.0)
                valids.append(0)
                details.append({
                    'pred': None,
                    'answer': ref,
                    'score': 0,
                    'correct': 0
                })
                continue

            valids.append(1)
            reward = multi_index_identification_reward(pred, ref)
            scores.append(reward)
            details.append({
                'pred': pred,
                'answer': ref,
                'score': reward,
                'correct': int(reward == 1.0)
            })

        n = len(scores)
        return {
            'score': sum(scores) / n * 100 if n > 0 else 0.0,
            'valid': sum(valids) / n * 100 if n > 0 else 0.0,
            'details': details,
        }


@ICL_EVALUATORS.register_module()
class MoleculariqGenerationEvaluator(BaseEvaluator):
    """Generation 任务评测：生成分子满足全部约束才得 1 分。

    输出指标：
      score  – 全约束满足准确率 (%)
      valid  – 合法 SMILES 比例 (%)
    """

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        scores, valids, details = [], [], []
        for pred_raw, ref in zip(predictions, references):
            pred = extract_json(pred_raw)
            if pred is None:
                scores.append(0.0)
                valids.append(0)
                details.append({
                    'pred': None,
                    'constraints': ref,
                    'score': 0,
                    'correct': 0
                })
                continue

            result = multi_constraint_generation_reward(pred,
                                                        ref,
                                                        return_details=True)
            reward = result.get('reward', 0.0)
            valid_smiles = result.get('valid_smiles', False)

            scores.append(reward)
            valids.append(1 if valid_smiles else 0)
            details.append({
                'pred': pred,
                'constraints': ref,
                'score': reward,
                'correct': int(reward == 1.0),
                'valid_smiles': valid_smiles,
            })

        n = len(scores)
        return {
            'score': sum(scores) / n * 100 if n > 0 else 0.0,
            'valid': sum(valids) / n * 100 if n > 0 else 0.0,
            'details': details,
        }
