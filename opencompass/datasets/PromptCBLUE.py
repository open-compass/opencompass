import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset  # 保持与 MMLUDataset 同级的导包风格


@LOAD_DATASET.register_module()
class PromptCBLUEDataset(BaseDataset):
    """Loader for PromptCBLUE life-science tasks (CHIP-CDN, CHIP-CTC …).

    - 只读 `dev.json`。
    - 保留指定 `task_dataset` 的所有任务类型（包括 normalization、cls 等）。
    - 若 `target` 不在 `answer_choices`，自动追加；并生成 `options_str`
      （形如 “A. 选项1\\nB. 选项2 …”）。
    - 返回 `DatasetDict`，将 dev 复制到 test 以满足评估流程。
    """

    @staticmethod
    def load(path: str, name: str, **kwargs):
        path = get_data_path(path)
        file_path = osp.join(path, 'dev.json')
        if not osp.exists(file_path):
            raise FileNotFoundError(f'`dev.json` not found under {path}')

        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                if rec.get('task_dataset') != name:
                    continue  # 过滤子数据集

                choices = rec.get('answer_choices', []).copy()
                target = rec.get('target')
                if target not in choices:
                    choices.append(target)

                options_str = '\n'.join(f'{chr(65+i)}. {opt}'
                                        for i, opt in enumerate(choices))

                records.append({
                    'input': rec['input'],
                    'answer_choices': choices,
                    'options_str': options_str,
                    'target': target,
                })

        # 保证列完整，即使 records 为空
        if records:
            ds = Dataset.from_list(records)
        else:
            ds = Dataset.from_dict({
                k: []
                for k in ['input', 'answer_choices', 'options_str', 'target']
            })
        dataset = DatasetDict(dev=ds, test=ds)  # dev 与 test 指向同一份
        return dataset
