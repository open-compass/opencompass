import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class ReasonBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        raw_data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                prompt = line.get('prompt', '')
                prompt_ppl = line.get('prompt_ppl', '')
                label = line.get('label', '')
                label_ppl = line.get('label_ppl', '')
                choices = line.get('choices', '')
                tag = line.get('tag', '')
                source = line.get('source', '')
                option_content = {choice: line[choice] for choice in choices}
                data = {
                    'prompt': prompt,
                    'label': label,
                    'prompt_ppl': prompt_ppl,
                    'label_ppl': str(label_ppl)[0],
                    'choices': choices,
                    'tag': tag,
                    'source': source,
                }
                data.update(option_content)
                raw_data.append(data)
        dataset = Dataset.from_list(raw_data)
        return dataset
