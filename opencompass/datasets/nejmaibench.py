import re

import pandas as pd
from datasets import Dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from .base import BaseDataset


def _parse(item, prompt_mode):
    # 1. 从 Choices 字符串里按行拆分出每个选项
    raw_choices = item.get('Choices', '')
    # 去掉首尾空白并按行分割，过滤掉空行
    lines = [
        line.strip() for line in raw_choices.strip().splitlines()
        if line.strip()
    ]

    # 2. 用正则去掉行首的 "A. "/"B. " 等前缀，只保留选项内容
    options_list = [re.sub(r'^[A-Z]\.\s*', '', line) for line in lines]

    # 3. 写回 item
    item['options'] = options_list

    # 4. 重建带标号的选项字符串
    options_str = '\n'.join(f'{chr(65 + i)}. {opt}'
                            for i, opt in enumerate(options_list))

    # 5. 构造 question、label、prompt_mode、start、end
    item['question'] = f"{item['Question']}\n{options_str}"
    item['label'] = item['Answer']
    item['prompt_mode'] = prompt_mode
    item['start'] = chr(65)
    item['end'] = chr(65 + len(options_list) - 1)
    return item


@LOAD_DATASET.register_module()
class NejmaibenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, prompt_mode: str = 'zero-shot', **kwargs):
        # 读取 CSV 文件为 DataFrame，并将 NaN 转为空字符串
        path = get_data_path(path)
        df = pd.read_csv(path, encoding='utf-8')
        df = df.fillna('')

        # 转换为字典列表
        data_list = df.to_dict(orient='records')

        # 将数据列表包装为 Dataset
        dataset = Dataset.from_list(data_list)

        # 根据提示模式进行解析
        if prompt_mode == 'zero-shot':
            dataset = dataset.map(lambda item: _parse(item, prompt_mode))
        elif prompt_mode == 'few-shot':
            pass  # TODO: Implement few-shot prompt handling
        return dataset


class NejmaibenchEvaluator(BaseEvaluator):

    def score(self, predictions, references, test_set):
        method = test_set['prompt_mode'][0]

        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        correct = 0
        count = 0
        details = []
        for idx, (i, j) in enumerate(zip(predictions, references)):
            i = answer_cleansing(method, i, test_set['options'][idx],
                                 test_set['label'][idx])
            detail = {
                'pred': i,
                'answer': j,
                'correct': False,
                'Subject': test_set['Subject'][idx],
            }
            count += 1
            if i == j:
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result


@TEXT_POSTPROCESSORS.register_module()
def answer_cleansing(
    method: str,
    prediction: str,
    options: list,
    label: str,
) -> str:

    # Clean up unwanted phrases in the prediction
    for unwanted_phrase in [
            'I understand',
            'A through J',
            'A through E',
            'A through D',
    ]:
        prediction = prediction.replace(unwanted_phrase, '')

    options_num = len(options)
    options = [chr(65 + i) for i in range(options_num)]
    options_str = r'\b(' + '|'.join(options) + r')\b'
    prediction = re.findall(options_str, prediction)

    if len(prediction) == 0:
        prediction = []
        return prediction
    else:
        # If there is a "label" and its length is 1,
        # process prediction accordingly
        if len(label) == 1:
            if method == 'few-shot':
                answer_flag = True if len(prediction) > 1 else False
                # choose the first or last element based on the answer_flag
                if answer_flag:
                    prediction = [prediction[0]]
                else:
                    prediction = [prediction[-1]]
            elif method == 'zero-shot':
                # choose the first element in list
                prediction = [prediction[0]]
            else:
                raise ValueError('Method is not properly defined ...')

            # Remove trailing period if it exists
            if prediction[0] and prediction[0].endswith('.'):
                prediction[0] = prediction[0][:-1]

        return prediction[0]
