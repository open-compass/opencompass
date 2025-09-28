import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class ChatMLDataset(BaseDataset):

    @staticmethod
    def load(path, file_name=None, local_mode=False):

        path = get_data_path(path, local_mode=local_mode)
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
        for item in data:
            user_content = next(
                (q['content']
                 for q in item['question'] if q['role'] == 'user'), None)
            first_answer = item['answer'][0] if item['answer'] else ''

            if user_content:
                extracted_data.append({
                    'extracted_question': user_content,
                    'extracted_answer': first_answer
                })

        data_final = Dataset.from_list(data)

        extracted_questions = [
            item['extracted_question'] for item in extracted_data
        ]
        extracted_answers = [
            item['extracted_answer'] for item in extracted_data
        ]

        # 将两列数据添加到原数据集
        data_final = data_final.add_column('extracted_question',
                                           extracted_questions)
        data_final = data_final.add_column('extracted_answer',
                                           extracted_answers)

        return data_final
