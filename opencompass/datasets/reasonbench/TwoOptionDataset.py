from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset
import json

@LOAD_DATASET.register_module()
class TwoOptionDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        raw_data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                prompt = line['prompt']
                prompt_ppl = line['prompt_ppl']
                label = line['label']
                label_ppl = line['label_ppl']
                choices = line['choices']
                tag = line['tag']
                source = line['source']
                A = line['A']
                B = line['B']
                raw_data.append({
                    'prompt': prompt,
                    'label': label,
                    'prompt_ppl': prompt_ppl,
                    'label_ppl': str(label_ppl)[0],
                    'choices': choices,
                    'tag': tag,
                    'source': source,
                    'A': A,
                    'B': B,
                })
        dataset = Dataset.from_list(raw_data)
        return dataset
