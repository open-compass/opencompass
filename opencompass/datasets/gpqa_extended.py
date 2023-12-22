import csv
import os
import copy

from datasets import Dataset, DatasetDict

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from .base import BaseDataset


@LOAD_DATASET.register_module()
class GPQAExtendedDataset(BaseDataset):

    @staticmethod
    def load(path):
        cnt = 0
        data = []
        data_new = []
        with open(os.path.join(path, 'gpqa_extended.csv'), 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[7] == 'Question':
                    continue
                cnt = cnt + 1
                question = row[7]
                A = row[8]
                B = row[9]
                C = row[10]
                D = row[11]
                options = [row[8], row[9], row[10], row[11]]
                answer = 'A'

                data.append({
                    'question': question,
                    'A': A,
                    'B': B,
                    'C': C,
                    'D': D,
                    'options': options,
                    'answer': answer
                })

                circular_patterns = ['ABCD', 'BCDA', 'CDAB', 'DABC']  # 更新选项顺序
                c = circular_patterns[cnt % 4]
                line = copy.deepcopy(data[cnt - 1])
                tmp = line['A']
                for i in range(4):
                    line['ABCD'[i]] = line['options'][ord(c[i]) - ord('A')]

                for i in range(4):
                    if line['ABCD'[i]] == tmp:
                        line['answer'] = 'ABCD'[i]
                        break
                data_new.append(line)

        dataset = Dataset.from_list(data_new)

        return dataset