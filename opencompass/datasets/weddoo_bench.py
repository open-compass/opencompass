import json
import os.path as osp
from typing import List

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class WeddooBenchDataset(BaseDataset):
    """WeddooBench 婚嫁垂类评测数据集加载器。

    数据格式（每行一个 JSON）：
    {
      "question": "30桌婚宴，餐标5888元/桌，15%服务费，总预算多少？",
      "choices": ["202,560元", "176,640元", "188,416元", "165,000元"],
      "answer": "A",
      "category": "budget_calculation",
      "difficulty": "medium"
    }
    """

    def load(self, path: str, reader_cfg: str = None) -> List[dict]:
        with open(path, 'r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f if line.strip()]

        data = []
        for line in lines:
            question = line['question']
            choices = line.get('choices', [])
            if choices:
                # 选择题：把选项拼进 prompt
                prompt = self._build_choice_prompt(question, choices)
                answer = line['answer']
            else:
                # 开放题
                prompt = question
                answer = line['answer']
            data.append({
                'prompt': prompt,
                'answer': answer,
                'category': line.get('category', 'general'),
                'difficulty': line.get('difficulty', 'medium'),
            })
        return data

    @staticmethod
    def _build_choice_prompt(question: str, choices: List[str]) -> str:
        options = ' '.join([f'{chr(65+i)}. {c}' for i, c in enumerate(choices)])
        return f"{question}\n选项：{options}\n请直接输出正确选项字母。"


class WeddooBenchEvaluator(BaseEvaluator):
    """WeddooBench  evaluator：支持选择题精确匹配和开放题关键词匹配。"""

    def score(self, predictions: List, references: List) -> dict:
        if len(predictions) != len(references):
            return {'error': 'predictions and references have different lengths'}

        correct = 0
        total = len(predictions)
        details = []

        for pred, ref in zip(predictions, references):
            pred_text = pred.strip() if isinstance(pred, str) else str(pred).strip()
            ref_text = ref.strip() if isinstance(ref, str) else str(ref).strip()

            # 选择题：答案通常是 A/B/C/D
            if len(ref_text) == 1 and ref_text in 'ABCD':
                is_correct = ref_text in pred_text[:10]
            else:
                # 开放题：简单关键词覆盖（可升级为 LLM-as-Judge）
                keywords = ref_text.split()
                match_count = sum(1 for k in keywords if k in pred_text)
                is_correct = match_count >= max(1, len(keywords) // 2)

            if is_correct:
                correct += 1
            details.append({'pred': pred_text, 'answer': ref_text, 'correct': is_correct})

        accuracy = 100 * correct / total if total else 0
        return {'accuracy': accuracy, 'correct': correct, 'total': total, 'details': details}
