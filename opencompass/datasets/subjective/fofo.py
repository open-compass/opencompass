# flake8: noqa
import json
import os.path as osp

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset

base_prompt = """
I would like you to create a leaderboard that evaluates the correctness of the format of answers from various large language models. To accomplish this, you will need to analyze the text prompts given to the models and their corresponding answers. Specifically, please ensure that your evaluation outputs are properly formatted as a json string. I will provide both the prompts and the responses for this purpose.

Here is the prompt:
{
    "instruction": "{instruction}",
}

Here are the outputs of the models:
[
    {
        "model": "model",
        "answer": "{prediction}"
    },
]

Please evaluate the formatting of the model's responses by checking if they comply with the format specifications stated in the prompt. Perform a thorough format check and provide a detailed explanation for why the format is correct or incorrect. Your feedback should include the name of the model, followed by the format correctness status represented as '1' for correct and '0' for incorrect. Present your reasoning as bullet points within a single string for each model assessed. In other words, you should produce the following output:
```json
[
    {
        'model': <model-name>,
        'format_correctness': <correctness>,
        'reasons': <reasons-of-format-correctness>
    }
]
```

Please note that your response should be a properly formatted JSON string and should not contain any additional content. We will load it directly as a JSON string in Python.
"""


@LOAD_DATASET.register_module()
class FofoDataset(BaseDataset):

    def load(self, path: str, name: str):
        filename = osp.join(path, f'{name}.json')
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for problem in json_data:
                question = problem['instruction']
                judge_prompt = base_prompt.replace("{instruction}", question)
                lan = 'cn' if 'cn' in name else 'en'
                raw_data.append({
                    'question': question,
                    'judge_prompt': judge_prompt,
                    'judge': {
                        'lan': lan,
                        'id': problem['id'],
                        'domain': problem['domain'],
                        'sub_domain': problem['sub_domain'],
                        'format': problem['format'],
                        'format_type': problem['format_type'],
                        'question': question
                    }
                })
        dataset = Dataset.from_list(raw_data)
        return dataset


