import os.path as osp
from typing import Dict, List, Union

from mmengine.config import ConfigDict


def model_abbr_from_cfg(cfg: Union[ConfigDict, List[ConfigDict]]) -> str:
    """Generate model abbreviation from the model's confg."""
    if isinstance(cfg, (list, tuple)):
        return '_'.join(model_abbr_from_cfg(c) for c in cfg)
    if 'abbr' in cfg:
        return cfg['abbr']
    model_abbr = cfg['type'] + '_' + '_'.join(
        osp.realpath(cfg['path']).split('/')[-2:])
    model_abbr = model_abbr.replace('/', '_')
    return model_abbr


def dataset_abbr_from_cfg(cfg: ConfigDict) -> str:
    """Returns dataset abbreviation from the dataset's confg."""
    if 'abbr' in cfg:
        return cfg['abbr']
    dataset_abbr = cfg['path']
    if 'name' in cfg:
        dataset_abbr += '_' + cfg['name']
    dataset_abbr = dataset_abbr.replace('/', '_')
    return dataset_abbr


def task_abbr_from_cfg(task: Dict) -> str:
    """Returns task abbreviation from the task's confg."""
    return '[' + ','.join([
        f'{model_abbr_from_cfg(model)}/'
        f'{dataset_abbr_from_cfg(dataset)}'
        for i, model in enumerate(task['models'])
        for dataset in task['datasets'][i]
    ]) + ']'


def get_infer_output_path(model_cfg: ConfigDict,
                          dataset_cfg: ConfigDict,
                          root_path: str = None,
                          file_extension: str = 'json') -> str:
    # TODO: Rename this func
    assert root_path is not None, 'default root_path is not allowed any more'
    model_abbr = model_abbr_from_cfg(model_cfg)
    dataset_abbr = dataset_abbr_from_cfg(dataset_cfg)
    return osp.join(root_path, model_abbr, f'{dataset_abbr}.{file_extension}')


def deal_with_judge_model_abbr(model_cfg, judge_model_cfg, meta=False):
    if isinstance(model_cfg, ConfigDict):
        model_cfg = (model_cfg, )
    if meta:
        for m_cfg in model_cfg:
            if 'summarized-by--' in m_cfg['abbr']:
                return model_cfg
        model_cfg += ({
            'abbr':
            'summarized-by--' + model_abbr_from_cfg(judge_model_cfg)
        }, )
    else:
        for m_cfg in model_cfg:
            if 'judged-by--' in m_cfg['abbr']:
                return model_cfg
        model_cfg += ({
            'abbr':
            'judged-by--' + model_abbr_from_cfg(judge_model_cfg)
        }, )
    return model_cfg
