from os import environ


def get_data_path(dataset_id: str, local_path: str):
    """return dataset id when getting data from ModelScope repo, otherwise just
    return local path as is."""
    if environ.get('DATASET_SOURCE') == 'ModelScope':
        return dataset_id

    return local_path
