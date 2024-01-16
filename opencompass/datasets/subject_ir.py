# flake8: noqa: E501
import json
import os.path as osp
import re
from typing import Optional

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .subjective_cmp import SubjectiveCmpDataset

eng_base_prefix = """
You are a judger. Please impartially judge whether an AI model's response to a question is correct based on the reference answers. You need to provide a conclusion of "correct" or "wrong," followed by the corresponding reasoning.

Note that since the reference answer is a candidate list, the AI model's response only needs to align with one item in the list to be deemed "correct."

Your judgment must strictly adhere to the following format:
Conclusion: [[Correct]]
Reasoning: xxx.

Conclusion: [[Wrong]]
Reasoning: xxx.

[Question Start]
{question}
[Question End]

[Reference Answers Start]
{ref}
[Reference Answers End]

[Model Response Start]
"""

chn_base_prefix = """
你是一个评判者，请你基于参考答案，公正地评判一个AI模型对于问题的回答是否正确。你需要给出“对或错”的结论，然后再给出相应的理由。
请注意，由于参考答案是一个候选列表，因此AI模型的回答只要符合列表中的某一项即可判断为“对”。
你的评判必须严格遵守以下格式：
结论：[[对]]
理由：xxx。

结论：[[错]]
理由：xxx。

[问题开始]
{question}
[问题结束]

[参考答案开始]
{ref}
[参考答案结束]

[模型回答开始]
"""


def prompt_construct(sample):
    lan = sample['others']['lan']
    question = sample['question']
    if lan == 'zh':
        prefix = chn_base_prefix.format(question=sample['question'],
                                        ref=str(sample['others']['answers']))
        suffix = '\n[模型回答结束]\n'
    elif lan == 'en':
        prefix = eng_base_prefix.format(question=sample['question'],
                                        ref=str(sample['others']['answers']))
        suffix = '\n[Model Response End]\n'
    return prefix, suffix


@LOAD_DATASET.register_module()
class IRDataset(SubjectiveCmpDataset):

    def load(
        self,
        path: str,
        name: str,
    ):
        dataset = list(super().load(path, name))
        subject_dataset = []
        for data in dataset:
            data['gpt4_prefix'], data['gpt4_suffix'] = prompt_construct(data)
            data['judge']['others'] = data['others']
            data['ref'] = str(data['others']['answers'])
            subject_dataset.append(data)
        dataset = Dataset.from_list(subject_dataset)
        return dataset
