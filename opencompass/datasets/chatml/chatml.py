# flake8: noqa
import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class ChatMLDataset(BaseDataset):
    """The Dataset class based on ChatML template that is only used to parse
    .jsonl files that conform to the following template format.

    {
        "question":[
            {
                "role": "system",
                "content": Str,
            },
            {
                "role": "user",
                "content": Str or List
                [
                    {
                        "type": Str, # "image"
                        "image_url": Str,
                    },
                    ...
                    {
                        "type": Str, # "text"
                        "text": Str,
                    },
                ]
            },
            {
                "role": "assistant",
                "content": Str
            },
            {
                "role": "user",
                "content": Str or List
            },
            ...
        ],
        "answer":[
            Str,
            Str,
            ...
        ]
    }
    {
        ...
    }
    ...

    Please use tools/chatml_format_test.py to check
    the format of your dataset files.

    """

    @staticmethod
    def load(path, file_name=None, local_mode=False):

        path = get_data_path(path, local_mode=True)
        with open(path, 'r', encoding='utf-8-sig') as f:
            data = [json.loads(line) for line in f]

        for i in range(len(data)):
            key_list = list(data[i].keys())
            for key in key_list:
                if key != 'question' and key != 'answer':
                    del data[i][key]

        from .verification import VerifyDataset
        for i in data:
            VerifyDataset(**i)

        input_prompt = '\nRemember to put your final answer within \\boxed{}.'
        for i in range(len(data)):
            for j in range(len(data[i]['question'])):
                if data[i]['question'][j]['role'] == 'user':
                    data[i]['question'][j]['content'] += input_prompt

        extracted_data = []
        data_final = Dataset.from_list(data)
        data_final = data_final.rename_column('question', 'chatml_question')
        data_final = data_final.rename_column('answer', 'chatml_answer')

        for item in data:
            user_content = next(
                (q['content']
                 for q in item['question'] if q['role'] == 'user'), None)
            first_answer = item['answer'][0] if item['answer'] else ''

            if user_content:
                extracted_data.append({
                    'question': user_content,
                    'answer': first_answer
                })

        extracted_questions = [item['question'] for item in extracted_data]
        extracted_answers = [item['answer'] for item in extracted_data]

        data_final = data_final.add_column('question', extracted_questions)
        data_final = data_final.add_column('answer', extracted_answers)

        return data_final
