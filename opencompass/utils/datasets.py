import os
from .fileio import download_and_extract_archive
from .datasets_info import DATASETS_MAPPING, DATASETS_URL
from .logging import get_logger


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
        
        if not os.path.exists(local_path):
            get_logger().info(f'{local_path} does not exist!'
                            'Start Download data automatically!'
                            'If you have downloaded the data before,'
                            'You can specific `COMPASS_DATA_CACHE` '
                            'to avoid downloading~')
            download_dataset()
        else:
            return local_path

def download_dataset(data_path, cache_dir, remove_finished=True):


    valid_data_names = list(DATASETS_URL.keys())
    dataset_name = ''
    for name in valid_data_names:
        if name in data_path:
            dataset_name = name
    assert dataset_name, f'No valid url for {data_path}!\n' + \
    f'Please make sure  `{data_path}` is correct'
    dataset_info = DATASETS_URL[dataset_name]
    dataset_url = dataset_info['url']
    dataset_md5 = dataset_info['md5']
    cache_dir = cache_dir if cache_dir else "~/.cache/opencompass/data"

    # download and extract files
    download_and_extract_archive(
        url=dataset_url, 
        download_root=cache_dir,
        md5=dataset_md5,
        remove_finished=remove_finished
    )

