import json
import os.path as osp
from typing import Optional

from datasets import Dataset, DatasetDict
from tqdm import trange

from opencompass.openicl.icl_retriever import BaseRetriever
from opencompass.utils import get_data_path

from .base import BaseDataset


class XiezhiDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path, local_mode=True)
        dataset = DatasetDict()
        filename = osp.join(path, name, 'xiezhi.v1.json')
        if 'chn' in name:
            train_filename = osp.join(path, 'xiezhi_train_chn',
                                      'xiezhi.v1.json')
        else:
            train_filename = osp.join(path, 'xiezhi_train_eng',
                                      'xiezhi.v1.json')
        for split, filename in [['train', train_filename], ['test', filename]]:
            raw_data = []
            with open(filename, encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if data['options'].endswith("\"\n"):
                        data['options'] = data['options'][:-2]
                    options = data['options'].split('\n')
                    if len(options) != 4:
                        continue
                    answer = 'ABCD'[options.index(data['answer'])]
                    # The longer the label, the more fine-grained the concept
                    labels = sorted(
                        data['labels' if 'chn' in name else 'label'],
                        key=lambda x: len(x),
                        reverse=True)
                    raw_data.append({
                        'question': data['question'],
                        'A': options[0],
                        'B': options[1],
                        'C': options[2],
                        'D': options[3],
                        'labels': labels,
                        'answer': answer,
                    })
            dataset[split] = Dataset.from_list(raw_data)
        return dataset


class XiezhiRetriever(BaseRetriever):

    def __init__(self,
                 dataset,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 ice_num: Optional[int] = 1) -> None:
        super().__init__(dataset, ice_separator, ice_eos_token, ice_num)

    def retrieve(self):
        """Retrieve in-context examples for each test case.

        For each one of the in-context example, there is a list of label,
        indicating the categories to which the example is related. For each one
        of the test case, there is also a list of label, indicating the
        categories. This retriever will retrieve the in-context examples that
        share at least one label with the test case.
        """
        label2indice = {}
        for index, item in enumerate(self.index_ds):
            for label in item['labels']:
                if label not in label2indice:
                    label2indice[label] = []
                label2indice[label].append(index)
        rtr_idx_list = []
        for index in trange(len(self.test_ds),
                            disable=not self.is_main_process):
            id_list = []
            for label in self.test_ds[index]['labels']:
                if len(id_list) < self.ice_num:
                    id_list += label2indice[label]
                else:
                    break
            rtr_idx_list.append(id_list[:self.ice_num])
        return rtr_idx_list
