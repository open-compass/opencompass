from datasets import load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class MMLUReduxDataset(BaseDataset):
    """Load the MMLU-Redux dataset and expand its choice list into columns.

    MMLU-Redux (https://github.com/aryopg/mmlu-redux) is a manually
    re-annotated subset of MMLU, loaded directly from the Hugging Face hub.
    Each example has `question` (str), `choices` (list of 4 strings) and
    `answer` (int index into the choices). The loader exposes `A`-`D` columns
    for prompt templating and a `target` column holding the correct letter.
    """

    @staticmethod
    def load(path: str, name: str, **kwargs):
        data = load_dataset(path, name, split='test', **kwargs)

        def _expand(example):
            return {
                'A': example['choices'][0],
                'B': example['choices'][1],
                'C': example['choices'][2],
                'D': example['choices'][3],
                'target': 'ABCD'[example['answer']],
            }

        data = data.map(_expand)
        return data
