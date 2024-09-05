#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import os
import re
from multiprocessing import Pool
from typing import List, Union

from mmengine.config import Config, ConfigDict


# from opencompass.utils import get_prompt_hash
# copied from opencompass.utils.get_prompt_hash, for easy use in ci
def get_prompt_hash(dataset_cfg: Union[ConfigDict, List[ConfigDict]]) -> str:
    """Get the hash of the prompt configuration.

    Args:
        dataset_cfg (ConfigDict or list[ConfigDict]): The dataset
            configuration.

    Returns:
        str: The hash of the prompt configuration.
    """
    if isinstance(dataset_cfg, list):
        if len(dataset_cfg) == 1:
            dataset_cfg = dataset_cfg[0]
        else:
            hashes = ','.join([get_prompt_hash(cfg) for cfg in dataset_cfg])
            hash_object = hashlib.sha256(hashes.encode())
            return hash_object.hexdigest()
    # for custom datasets
    if 'infer_cfg' not in dataset_cfg:
        dataset_cfg.pop('abbr', '')
        dataset_cfg.pop('path', '')
        d_json = json.dumps(dataset_cfg.to_dict(), sort_keys=True)
        hash_object = hashlib.sha256(d_json.encode())
        return hash_object.hexdigest()
    # for regular datasets
    if 'reader_cfg' in dataset_cfg.infer_cfg:
        # new config
        reader_cfg = dict(type='DatasetReader',
                          input_columns=dataset_cfg.reader_cfg.input_columns,
                          output_column=dataset_cfg.reader_cfg.output_column)
        dataset_cfg.infer_cfg.reader = reader_cfg
        if 'train_split' in dataset_cfg.infer_cfg.reader_cfg:
            dataset_cfg.infer_cfg.retriever[
                'index_split'] = dataset_cfg.infer_cfg['reader_cfg'][
                    'train_split']
        if 'test_split' in dataset_cfg.infer_cfg.reader_cfg:
            dataset_cfg.infer_cfg.retriever[
                'test_split'] = dataset_cfg.infer_cfg.reader_cfg.test_split
        for k, v in dataset_cfg.infer_cfg.items():
            dataset_cfg.infer_cfg[k]['type'] = v['type'].split('.')[-1]
    # A compromise for the hash consistency
    if 'fix_id_list' in dataset_cfg.infer_cfg.retriever:
        fix_id_list = dataset_cfg.infer_cfg.retriever.pop('fix_id_list')
        dataset_cfg.infer_cfg.inferencer['fix_id_list'] = fix_id_list
    d_json = json.dumps(dataset_cfg.infer_cfg.to_dict(), sort_keys=True)
    hash_object = hashlib.sha256(d_json.encode())
    return hash_object.hexdigest()


# Assuming get_hash is a function that computes the hash of a file
# from get_hash import get_hash
def get_hash(path):
    cfg = Config.fromfile(path)
    for k in cfg.keys():
        if k.endswith('_datasets'):
            return get_prompt_hash(cfg[k])[:6]
    print(f'Could not find *_datasets in {path}')
    return None


def check_and_rename(filepath):
    base_name = os.path.basename(filepath)
    match = re.match(r'(.*)_(gen|ppl|ll|mixed)_(.*).py', base_name)
    if match:
        dataset, mode, old_hash = match.groups()
        new_hash = get_hash(filepath)
        if not new_hash:
            return None, None
        if old_hash != new_hash:
            new_name = f'{dataset}_{mode}_{new_hash}.py'
            new_file = os.path.join(os.path.dirname(filepath), new_name)
            print(f'Rename {filepath} to {new_file}')
            return filepath, new_file
    return None, None


def update_imports(data):
    python_file, name_pairs = data
    for filepath, new_file in name_pairs:
        old_name = os.path.basename(filepath)[:-3]
        new_name = os.path.basename(new_file)[:-3]
        if not os.path.exists(python_file):
            return
        with open(python_file, 'r') as file:
            filedata = file.read()
        # Replace the old name with new name
        new_data = filedata.replace(old_name, new_name)
        if filedata != new_data:
            with open(python_file, 'w') as file:
                file.write(new_data)
            # print(f"Updated imports in {python_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('python_files', nargs='*')
    # Could be opencompass/configs/datasets and configs/datasets
    parser.add_argument('--root_folder', default='configs/datasets')
    args = parser.parse_args()

    root_folder = args.root_folder
    if args.python_files:
        python_files = [
            i for i in args.python_files if i.startswith(root_folder)
        ]
    else:
        python_files = glob.glob(f'{root_folder}/**/*.py', recursive=True)

    # Use multiprocessing to speed up the check and rename process
    with Pool(16) as p:
        name_pairs = p.map(check_and_rename, python_files)
    name_pairs = [pair for pair in name_pairs if pair[0] is not None]
    if not name_pairs:
        return
    with Pool(16) as p:
        p.starmap(os.rename, name_pairs)
    root_folder = 'configs'
    python_files = glob.glob(f'{root_folder}/**/*.py', recursive=True)
    update_data = [(python_file, name_pairs) for python_file in python_files]
    with Pool(16) as p:
        p.map(update_imports, update_data)


if __name__ == '__main__':
    main()
