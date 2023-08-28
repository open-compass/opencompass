#!/usr/bin/env python3
import glob
import os
import re
from multiprocessing import Pool

from mmengine.config import Config

from opencompass.utils import get_prompt_hash


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
    match = re.match(r'(.*)_(gen|ppl)_(.*).py', base_name)
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
        with open(python_file, 'r') as file:
            filedata = file.read()
        # Replace the old name with new name
        new_data = filedata.replace(old_name, new_name)
        if filedata != new_data:
            with open(python_file, 'w') as file:
                file.write(new_data)
            # print(f"Updated imports in {python_file}")


def main():
    root_folder = 'configs/datasets'
    python_files = glob.glob(f'{root_folder}/**/*.py', recursive=True)
    # Use multiprocessing to speed up the check and rename process
    with Pool(16) as p:
        name_pairs = p.map(check_and_rename, python_files)
    name_pairs = [pair for pair in name_pairs if pair[0] is not None]
    with Pool(16) as p:
        p.starmap(os.rename, name_pairs)
    python_files = glob.glob(f'{root_folder}/**/*.py', recursive=True)
    update_data = [(python_file, name_pairs) for python_file in python_files]
    with Pool(16) as p:
        p.map(update_imports, update_data)


if __name__ == '__main__':
    main()
