import importlib
import os
import sys
from contextlib import contextmanager

_VLMEVAL_INSTALL = 'Install VLMEvalKit so `import vlmeval` works.'


def _load_build_dataset():
    if sys.version_info < (3, 10):
        raise RuntimeError(
            'The VLMEvalKit vlm extra requires Python 3.10 or newer.')
    try:
        module = importlib.import_module('vlmeval.dataset')
    except ModuleNotFoundError as e:
        if e.name == 'vlmeval':
            raise ModuleNotFoundError(
                'VLMEvalKit is required for this dataset. '
                f'{_VLMEVAL_INSTALL}') from e
        if e.name and e.name.startswith('vlmeval.'):
            raise ModuleNotFoundError(
                'The installed VLMEvalKit is incomplete or missing optional '
                f'dependencies. {_VLMEVAL_INSTALL}') from e
        raise
    return module.build_dataset


@contextmanager
def _lmu_data_root(data_root):
    if data_root is None:
        yield
        return
    data_root = os.path.abspath(os.path.expanduser(data_root))
    os.makedirs(data_root, exist_ok=True)
    old_root = os.environ.get('LMUData')
    os.environ['LMUData'] = data_root
    try:
        yield
    finally:
        if old_root is None:
            os.environ.pop('LMUData', None)
        else:
            os.environ['LMUData'] = old_root


def build_vlmeval_dataset(dataset_name, data_root=None, **kwargs):
    with _lmu_data_root(data_root):
        dataset = _load_build_dataset()(dataset_name, **kwargs)
    if dataset is None:
        raise ValueError(
            f'VLMEvalKit failed to build dataset {dataset_name!r}.')
    modality = getattr(dataset, 'MODALITY', None)
    if modality != 'IMAGE':
        raise NotImplementedError(
            f'VLMEvalKit dataset {dataset_name!r} has modality {modality!r}; '
            'this OpenCompass bridge currently supports IMAGE datasets only.')
    if getattr(dataset, 'TYPE', None) == 'MT':
        raise NotImplementedError(
            f'VLMEvalKit dataset {dataset_name!r} requires multi-turn '
            'inference, which is not supported by this single-turn bridge.')
    data = getattr(dataset, 'data', None)
    if data is None or 'index' not in data:
        raise ValueError(
            f'VLMEvalKit dataset {dataset_name!r} has no `index` column.')
    if data['index'].isna().any():
        raise ValueError(
            f'VLMEvalKit dataset {dataset_name!r} has null `index` values.')
    if data['index'].map(str).duplicated().any():
        raise ValueError(
            f'VLMEvalKit dataset {dataset_name!r} has duplicate `index` '
            'values.')
    return dataset
