import concurrent.futures
import json
import os
import os.path as osp
import re
import subprocess
import sys

import h5py
import numpy as np
import scipy
import scipy.sparse
import sympy
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class SciCodeDataset(BaseDataset):

    @staticmethod
    def load(path, with_bg, **kwargs):
        test_data = []
        path = get_data_path(path, local_mode=True)
        if with_bg:  # test with background
            file_path = osp.join(path, 'SciCode_datasets_with_background.json')
        else:  # test w/o background
            file_path = osp.join(path, 'SciCode_datasets.json')

        with open(file_path, 'r', encoding='utf-8') as file:
            test_data = json.load(file)

        dataset = Dataset.from_list(test_data)
        return dataset

    def return_dataset(self):
        return self.dataset


def process_hdf5_list(group):
    lst = []
    for key in group.keys():
        lst.append(group[key][()])
    return lst


def process_hdf5_dict(group):
    dict = {}
    for key, obj in group.items():
        if isinstance(obj, h5py.Group):
            dict[key] = process_hdf5_sparse_matrix(obj['sparse_matrix'])
        elif isinstance(obj[()], bytes):
            dict[key] = obj[()].decode('utf-8', errors='strict')
        else:
            try:
                tmp = float(key)
                dict[tmp] = obj[()]
            except ValueError:
                dict[key] = obj[()]
    return dict


def process_hdf5_sparse_matrix(group):
    data = group['data'][()]
    shape = tuple(group['shape'][()])
    if 'row' in group and 'col' in group:
        row = group['row'][()]
        col = group['col'][()]
        return scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
    elif 'blocksize' in group:
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        blocksize = tuple(group['blocksize'][()])
        return scipy.sparse.bsr_matrix((data, indices, indptr),
                                       shape=shape,
                                       blocksize=blocksize)
    else:
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)


def process_hdf5_datagroup(group):
    for key in group.keys():
        if key == 'list':
            return process_hdf5_list(group[key])
        if key == 'sparse_matrix':
            return process_hdf5_sparse_matrix(group[key])
        else:
            return process_hdf5_dict(group)


def process_hdf5_to_tuple(step_id, test_num):

    H5PY_FILE_FOLDER = './data/scicode/test_data'
    H5PY_FILE_FOLDER = get_data_path(H5PY_FILE_FOLDER, local_mode=True)
    data_lst = []
    H5PY_FILE = os.path.join(H5PY_FILE_FOLDER, f'{step_id}.h5')
    assert os.path.exists(
        H5PY_FILE
    ), f"Please manually download 'test_data.h5' from https://github.com/open-compass/storage/releases/download/v0.1.0/scicode_test_data.zip and put the file in {H5PY_FILE}"  # noqa: E501

    with h5py.File(H5PY_FILE, 'r') as f:
        for test_id in range(test_num):
            group_path = f'{step_id}/test{test_id + 1}'
            if isinstance(f[group_path], h5py.Group):
                group = f[group_path]  # test1, test2, test3
                num_keys = [key for key in group.keys()]
                if len(num_keys) == 1:  # only 1 var in the test
                    subgroup = group[num_keys[0]]
                    if isinstance(subgroup, h5py.Dataset):
                        if isinstance(subgroup[()], bytes):
                            data_lst.append(subgroup[()].decode(
                                'utf-8', errors='strict'))
                        else:
                            data_lst.append(subgroup[()])
                    elif isinstance(subgroup, h5py.Group):
                        data_lst.append(process_hdf5_datagroup(subgroup))
                else:
                    var_lst = []
                    for key in group.keys():  # var1, var2, var3
                        subgroup = group[key]
                        if isinstance(subgroup, h5py.Dataset):
                            if isinstance(subgroup[()], bytes):
                                var_lst.append(subgroup[()].decode(
                                    'utf-8', errors='strict'))
                            else:
                                var_lst.append(subgroup[()])
                        elif isinstance(subgroup, h5py.Group):
                            var_lst.append(process_hdf5_datagroup(subgroup))
                    data_lst.append(tuple(var_lst))
            else:
                raise FileNotFoundError(
                    f'Path {group_path} not found in the file.')
    return data_lst


def are_dicts_close(dict1, dict2, atol=1e-8, rtol=1e-5):
    dict1 = process_symbol_in_dict(dict1)
    dict2 = process_symbol_in_dict(dict2)
    # Check if both dictionaries have the same keys
    if dict1.keys() != dict2.keys():
        return False

    # Check if the corresponding values are close
    for key in dict1:
        value1 = dict1[key]
        value2 = dict2[key]
        if isinstance(value1, (sympy.Symbol, str)):
            if not value1 == value2:
                return False
        elif isinstance(value1,
                        (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix,
                         scipy.sparse.bsr_matrix, scipy.sparse.coo_matrix)):
            value1 = value1.toarray()
            value2 = value2.toarray()
            if not np.allclose(value1, value2, atol=atol, rtol=rtol):
                return False
        # Use np.allclose to compare values
        else:
            try:
                if not np.allclose(value1, value2, atol=atol, rtol=rtol):
                    return False
            except ValueError:
                if not value1 == value2:
                    return False

    return True


def process_symbol_in_dict(dict):
    new_dict = {}
    for key, value in dict.items():
        new_dict[key] = value
        if isinstance(value, sympy.Symbol):
            new_dict[key] = str(value)
        if isinstance(key, sympy.Symbol):
            new_dict[str(key)] = dict[key]
            new_dict.pop(key)
    return new_dict


def are_csc_matrix_close(matrix1, matrix2):
    dense1 = matrix1.toarray()
    dense2 = matrix2.toarray()
    return np.allclose(dense1, dense2)


def cmp_tuple_or_list(var1, var2):
    if len(var1) != len(var2):
        return False
    for v1, v2 in zip(var1, var2):
        if isinstance(v1, dict):
            if not are_dicts_close(v1, v2):
                return False
        elif isinstance(v1,
                        (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
            if not are_csc_matrix_close(v1, v2):
                return False
        elif isinstance(v1, bool):
            if not v1 == v2:
                return False
        else:
            try:
                if not np.allclose(v1, v2):
                    return False
            except ValueError as e:
                print(e)
                if not v1 == v2:
                    return False
    return True


@ICL_EVALUATORS.register_module()
class SciCodeEvaluator(BaseEvaluator):

    def __init__(self, dataset_path, with_bg):
        super().__init__()
        test_data = []
        dataset_path = get_data_path(dataset_path, local_mode=True)
        if with_bg:  # test with background
            file_path = osp.join(dataset_path,
                                 'SciCode_datasets_with_background.json')
        else:  # test w/o background
            file_path = osp.join(dataset_path, 'SciCode_datasets.json')
        with open(file_path, 'r', encoding='utf-8') as file:
            test_data = json.load(file)
        self.dataset = Dataset.from_list(test_data)

    def extract_python_script(self, response: str):
        start_marker = '```python'
        end_marker = '```'

        if start_marker not in response or end_marker not in response:
            # If the markers are not present, return an empty string
            # print("fail to follow the instruct")
            return ''

        # Split the response at the start marker and take the second part
        after_start = response.split(start_marker)
        if len(after_start) < 2:
            return ''  # No valid split was made

        # Split the part after the start marker at the end marker
        python_script = after_start[1].split(end_marker)[0]

        # Remove leading import statements using regex
        python_script = re.sub(r'^\s*(import .*|from .*\s+import\s+.*)',
                               '',
                               python_script,
                               flags=re.MULTILINE)

        return python_script

    def run_script(self, script_path):
        try:
            subprocess.run([sys.executable, script_path],
                           check=True,
                           capture_output=True,
                           text=True,
                           timeout=120)
            return 0
        except subprocess.CalledProcessError:
            return 1
        except subprocess.TimeoutExpired:
            return 2

    def score(self, predictions, references):
        # generate all python test codes
        for idx, prediction_list in enumerate(predictions):
            # traverse each test sample
            problem_id = self.dataset[idx]['id']
            num_of_subproblems = len(prediction_list)

            # create dir for each test sample
            testdir_path = os.path.join(self._out_dir, str(problem_id))
            os.makedirs(testdir_path, exist_ok=True)

            python_code = ''
            # add import statement
            python_code += self.dataset[idx]['import']

            for sub_idx in range(num_of_subproblems):
                # extract code
                response = prediction_list[sub_idx]
                python_code += self.extract_python_script(response)

                # process special examples
                if problem_id == '13' and sub_idx >= 5 or \
                   problem_id == '62' and sub_idx >= 0 or \
                   problem_id == '76' and sub_idx >= 2:
                    sub_idx += 1

                # test cases
                test_lst = self.dataset[idx]['test'][sub_idx]

                testfile_path = os.path.join(testdir_path,
                                             f'{problem_id}-{sub_idx + 1}.py')
                # write python code and test cases to a real python file
                with open(testfile_path, 'w', encoding='utf-8') as f:
                    f.write(python_code)
                    f.write("""

from opencompass.datasets.scicode import process_hdf5_to_tuple

""")
                    f.write('targets = process_hdf5_to_tuple(' +
                            f"'{problem_id}.{sub_idx + 1}', {len(test_lst)})" +
                            '\n')
                    for idx2 in range(len(test_lst)):
                        f.write(f'target = targets[{idx2}]\n\n')
                        for line in test_lst[idx2].split('\n'):
                            f.write(line + '\n')

        # find all scripts
        python_scripts = []
        for root, dirs, files in os.walk(self._out_dir):
            for file in files:
                if file.endswith('.py'):
                    python_scripts.append(os.path.join(root, file))

        # Use ThreadPoolExecutor to concurrently execute scripts
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit task and obtain Future object
            futures = [
                executor.submit(self.run_script, script)
                for script in python_scripts
            ]

        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)

        all_results = {}
        for script_path, result in zip(python_scripts, results):
            basename = os.path.basename(script_path)
            main_id = basename.split('-')[0]
            if all_results.get(main_id):
                all_results[main_id].append(result)
            else:
                all_results[main_id] = [result]

        correct, sub_correct = 0, 0
        count, sub_count = 0, 0

        for main_id in all_results:
            correct += sum(all_results[main_id]) == 0
            count += 1
            for sub in all_results[main_id]:
                sub_correct += sub == 0
                sub_count += 1

        result = {
            'accuracy': 100 * correct / count,
            'sub_accuracy': 100 * sub_correct / sub_count,
        }
        return result
