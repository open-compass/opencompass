import json
import os
from typing import List

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

DEFAULT_FILENAME = 'weddoo_bench_v1.jsonl'


def _format_key_points(key_points: List[str]) -> str:
    """Render key points as a numbered list for the judge prompt."""
    if not key_points:
        return '（无）'
    return '\n'.join(f'{i + 1}. {p}' for i, p in enumerate(key_points))


@LOAD_DATASET.register_module()
class WeddooBenchDataset(BaseDataset):
    """WeddooBench: an open-ended benchmark for the Chinese wedding industry.

    Every line of the source file is a JSON object::

        {
          "id": "WB-001",
          "question": "杭州有哪些比较好的婚礼堂？能推荐几家吗？",
          "answer": "参考答案……",
          "key_points": ["要点一", "要点二"],
          "category": "场地知识",
          "difficulty": "medium"
        }

    The dataset is evaluated with an LLM judge (see
    ``configs/datasets/weddoo_bench``), so ``key_points`` is kept as a
    separate column and passed to the judge as grading criteria.
    """

    @staticmethod
    def load(path: str, **kwargs) -> Dataset:

        path = get_data_path(path)
        if os.path.isdir(path):
            path = os.path.join(path, DEFAULT_FILENAME)

        data = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                key_points = item.get('key_points') or []
                data.append({
                    'id': item.get('id', ''),
                    'question': item['question'],
                    'answer': item['answer'],
                    'key_points': key_points,
                    'key_points_text': _format_key_points(key_points),
                    'category': item.get('category', 'general'),
                    'difficulty': item.get('difficulty', 'medium'),
                })
        if not data:
            raise ValueError(f'No samples found in {path}')
        return Dataset.from_list(data)
