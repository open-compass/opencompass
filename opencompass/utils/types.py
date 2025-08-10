from typing import Any, Dict, List, Union

from datasets import Dataset, DatasetDict
from mmengine.config import Config

from opencompass.registry import TASKS


def get_type_from_cfg(cfg: Union[Config, Dict]) -> Any:
    """Get the object type given MMEngine's Config.

    It loads the "type" field and return the corresponding object type.
    """
    type = cfg['type']
    if isinstance(type, str):
        # FIXME: This has nothing to do with any specific registry, to be fixed
        # in MMEngine
        type = TASKS.get(type)
    return type


def _check_type_list(obj, typelist: List):
    for _type in typelist:
        if _type is None:
            if obj is None:
                return obj
        elif isinstance(obj, _type):
            return obj
    raise TypeError(
        f'Expected an object in {[_.__name__ if _ is not None else None for _ in typelist]} type, but got {obj}'  # noqa
    )


def _check_dataset(obj) -> Union[Dataset, DatasetDict]:
    if isinstance(obj, Dataset) or isinstance(obj, DatasetDict):
        return obj
    else:
        raise TypeError(
            f'Expected a datasets.Dataset or a datasets.DatasetDict object, but got {obj}'  # noqa
        )


def _check_list(obj) -> List:
    if isinstance(obj, List):
        return obj
    else:
        raise TypeError(f'Expected a List object, but got {obj}')


def _check_str(obj) -> str:
    if isinstance(obj, str):
        return obj
    else:
        raise TypeError(f'Expected a str object, but got {obj}')


def _check_dict(obj) -> Dict:
    if isinstance(obj, Dict):
        return obj
    else:
        raise TypeError(f'Expected a Dict object, but got {obj}')
