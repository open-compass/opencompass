from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset  # 保持与 MMLUDataset 同级的导包风格


@LOAD_DATASET.register_module()
class PromptCBLUEDataset(BaseDataset):
    """Loader for PromptCBLUE life-science tasks (CHIP-CDN, CHIP-CTC …).

    - 只读 validation split。
    - 保留指定 `task_dataset` 的所有任务类型。
    - 若 `target` 不在 `answer_choices`，自动追加；并生成 `options_str`。
    - 返回 `DatasetDict`，包含 `validation` 和 `test`，以满足评估流程。
    """

    @staticmethod
    def load(path: str, name: str, **kwargs):
        # 1) 从 HuggingFace 读取 validation split
        hf_ds = load_dataset(path, split='validation', **kwargs)

        # 2) 过滤子数据集并构造记录
        records = []
        for rec in hf_ds:
            if rec.get('task_dataset') != name:
                continue

            choices = rec.get('answer_choices', []).copy()
            target = rec.get('target')
            if target not in choices:
                choices.append(target)

            options_str = '\n'.join(f'{chr(65 + i)}. {opt}'
                                    for i, opt in enumerate(choices))

            records.append({
                'input': rec['input'],
                'answer_choices': choices,
                'options_str': options_str,
                'target': target,
            })

        # 3) 构造 Dataset
        if records:
            validation_ds = Dataset.from_list(records)
        else:
            validation_ds = Dataset.from_dict({
                k: []
                for k in [
                    'input',
                    'answer_choices',
                    'options_str',
                    'target',
                ]
            })

        # 4) 返回时包含 validation 和 test
        return DatasetDict(
            validation=validation_ds,
            test=validation_ds,
        )
