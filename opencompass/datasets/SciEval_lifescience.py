import re
from typing import List

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET

# 预编译的多选题正则，按 PEP-8 每行 < 79 字符
_PATTERN_MC = (
    r'^(?P<stem>.*?)'  # 题干
    r'(?:A\.)\s*(?P<A>.*?)\s*'  # 选项 A
    r'B\.\s*(?P<B>.*?)\s*'  # 选项 B
    r'C\.\s*(?P<C>.*?)\s*'  # 选项 C
    r'D\.\s*(?P<D>.*?)'  # 选项 D
    r'Answer:'  # 答案分隔符
)


@LOAD_DATASET.register_module()
class SciEvalDataset(BaseDataset):
    """Biology multiple-choice subset of SciEval."""

    @staticmethod
    def load(path: str, name: str, **kwargs) -> DatasetDict:
        dataset = DatasetDict()

        for split in ('test', ):
            raw_iter = load_dataset(
                path,
                name=name,
                split=split,
                streaming=True,
            )

            examples: List[dict] = []
            for ex in raw_iter:
                if (ex.get('category') != 'biology'
                        or ex.get('type') != 'multiple-choice'):
                    continue

                ans_list = ex.get('answer') or ex.get('answers') or []
                if not ans_list:
                    continue
                target = ans_list[0]

                match = re.search(_PATTERN_MC, ex.get('question', ''), re.S)
                if not match:
                    continue

                examples.append({
                    'input': match.group('stem').strip(),
                    'A': match.group('A').strip(),
                    'B': match.group('B').strip(),
                    'C': match.group('C').strip(),
                    'D': match.group('D').strip(),
                    'target': target,
                })

            dataset[split] = Dataset.from_list(examples)

        return dataset
