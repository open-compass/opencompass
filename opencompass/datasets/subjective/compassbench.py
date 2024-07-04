# flake8: noqa
import json
import os.path as osp

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset

base_prompt_zh = """请根据 用户问题 以及 相应的两个回答，判断哪一个回答更好。
[用户问题]
{question}

[回答1开始]
{prediction}
[回答1结束]

[回答2开始]
{prediction2}
[回答2结束]

请先对两个回答进行评价，最后在以下 3 个选项中做出选择:
A. 回答1更好
B. 回答2更好
C. 回答1、2平局

如果你认为回答1更好，你的输出应形如：
评价1：回答1 xxx
评价2：回答2 xxx
选择：[[A]]

如果你认为回答2更好，你的输出应形如：
评价1：回答1 xxx
评价2：回答2 xxx
选择：[[B]]

如果你认为回答1、2打成平手，你的输出应形如：
评价1：回答1 xxx
评价2：回答2 xxx
选择：[[C]]
"""

base_prompt_en = """Please evaluate the two responses based on the user's question and then choose from the following three options:
A. Response 1 is better
B. Response 2 is better
C. Both responses are equal

[user's question]
{question}

[Response 1 Start]
{prediction}
[Response 1 End]

[Response 2 Start]
{prediction2}
[Response 2 End]

If you believe that Response 1 is better, your output should be formatted as follows:
Evaluation 1: Response 1 xxx
Evaluation 2: Response 2 xxx
Choice: [[A]]

If you believe that Response 2 is better, your output should be formatted as follows:
Evaluation 1: Response 1 xxx
Evaluation 2: Response 2 xxx
Choice: [[B]]

If you believe that both responses are equally good, your output should be formatted as follows:
Evaluation 1: Response 1 xxx
Evaluation 2: Response 2 xxx
Choice: [[C]]
"""


@LOAD_DATASET.register_module()
class CompassBenchDataset(BaseDataset):

    def load(self, path: str, name: str, *args, **kwargs):
        filename = osp.join(path, f'{name}.json')
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for problem in json_data:
                question = problem['question']
                lan = problem['language']
                others = problem['others']
                judge_prompt = base_prompt_zh if lan == 'zh' else base_prompt_en
                judge_prompt = judge_prompt.replace('{question}', question)
                raw_data.append({
                    'question': question,
                    'judge_prompt': judge_prompt,
                    'judge': {
                        'lan': lan,
                        'level': others['level'],
                        'category': problem['category'],
                        'question': question
                    }
                })
        dataset = Dataset.from_list(raw_data)
        return dataset
