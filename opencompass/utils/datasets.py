import os
from .datasets_info import DATASETS_MAPPING, DATASETS_URL


def get_data_path(dataset_id: str, local_mode: bool = False):
    """return dataset id when getting data from ModelScope repo, otherwise just
    return local path as is.

    Args:
        dataset_id (str): dataset id or data path
        local_mode (bool): whether to use local path or
            ModelScope/HuggignFace repo
    """
    # update the path with CACHE_DIR
    cache_dir = os.environ.get('COMPASS_DATA_CACHE', '')

    # For absolute path customized by the users
    if dataset_id.startswith('/'):
        return dataset_id

    # For relative path, with CACHE_DIR
    if local_mode:
        local_path = os.path.join(cache_dir, dataset_id)
        assert os.path.exists(local_path), f'{local_path} does not exist!'
        return local_path

    dataset_source = os.environ.get('DATASET_SOURCE', None)
    if dataset_source == 'ModelScope':
        ms_id = DATASETS_MAPPING[dataset_id]['ms_id']
        assert ms_id is not None, \
            f'{dataset_id} is not supported in ModelScope'
        return ms_id
    elif dataset_source == 'HF':
        # TODO: HuggingFace mode is currently not supported!
        hf_id = DATASETS_MAPPING[dataset_id]['hf_id']
        assert hf_id is not None, \
            f'{dataset_id} is not supported in HuggingFace'
        return hf_id
    else:
        # for the local path
        local_path = DATASETS_MAPPING[dataset_id]['local']
        local_path = os.path.join(cache_dir, local_path)
        assert os.path.exists(local_path), f'{local_path} does not exist!'
        return local_path
