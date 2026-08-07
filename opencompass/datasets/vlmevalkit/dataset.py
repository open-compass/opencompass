import json

import pandas as pd
from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset
from .builder import _lmu_data_root, build_vlmeval_dataset
from .image import convert_vlmeval_prompt


def make_vlmeval_sample_id(dataset_name, index):
    return f'{dataset_name}:{index}'


def _python_value(value):
    if isinstance(value, dict):
        return {str(k): _python_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_python_value(v) for v in value]
    if hasattr(value, 'item') and not isinstance(value, (str, bytes)):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


@LOAD_DATASET.register_module()
class VLMEvalKitDataset(BaseDataset):

    def __init__(self, reader_cfg=None, **kwargs):
        reader_cfg = dict(reader_cfg or {})
        test_range = reader_cfg.get('test_range')
        if isinstance(test_range, str):
            reader_cfg.pop('test_range')
            kwargs['test_range'] = test_range
        super().__init__(reader_cfg=reader_cfg, **kwargs)

    @staticmethod
    def load(dataset_name='MMBench_DEV_EN',
             data_root=None,
             dataset_kwargs=None,
             sample_limit=None,
             test_range=None):
        if sample_limit is not None and sample_limit <= 0:
            raise ValueError('VLMEvalKit sample_limit must be positive.')
        dataset_kwargs = dict(dataset_kwargs or {})
        vlm_dataset = build_vlmeval_dataset(dataset_name, data_root,
                                            **dataset_kwargs)
        rows = []
        sample_ids = set()
        data = vlm_dataset.data
        if sample_limit is not None:
            data = data.iloc[:sample_limit]
        if test_range:
            scope = {'index_list': list(range(len(data)))}
            data = data.iloc[eval(f'index_list{test_range}', scope)]
        with _lmu_data_root(data_root):
            for _, row in data.iterrows():
                raw_sample = {
                    str(k): _python_value(v)
                    for k, v in row.to_dict().items() if k != 'image'
                }
                index = raw_sample['index']
                sample_id = make_vlmeval_sample_id(dataset_name, index)
                if sample_id in sample_ids:
                    raise ValueError(
                        f'Duplicate VLMEvalKit sample ID: {sample_id}')
                sample_ids.add(sample_id)
                metadata = {
                    k: raw_sample[k]
                    for k in ('split', 'category', 'l2-category', 'source',
                              'subject', 'topic_difficulty', 'id')
                    if k in raw_sample
                }
                rows.append(
                    dict(sample_id=sample_id,
                         prompt=convert_vlmeval_prompt(
                             vlm_dataset.build_prompt(row)),
                         vlmeval_index=index,
                         dataset_name=dataset_name,
                         raw_sample=json.dumps(raw_sample,
                                               ensure_ascii=False,
                                               sort_keys=True,
                                               default=str),
                         metadata=metadata,
                         answer=raw_sample.get('answer')))
        return Dataset.from_list(rows)
