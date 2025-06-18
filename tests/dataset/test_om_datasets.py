import random
import sys
import unittest
import warnings
from os import environ

from datasets import Dataset, DatasetDict
from mmengine.config import read_base
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore', category=DeprecationWarning)


def reload_datasets():
    modules_to_remove = [
        module_name for module_name in sys.modules
        if module_name.startswith('configs.datasets')
    ]

    for module_name in modules_to_remove:
        del sys.modules[module_name]

    with read_base():
        from configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets

    return sum((v for k, v in locals().items() if k.endswith('_datasets')), [])


def load_datasets_conf(source):
    environ['DATASET_SOURCE'] = source
    datasets_conf = reload_datasets()
    return datasets_conf


def load_datasets(source, conf):
    environ['DATASET_SOURCE'] = source
    if 'lang' in conf:
        dataset = conf['type'].load(path=conf['path'], lang=conf['lang'])
        return dataset

    if 'setting_name' in conf:
        dataset = conf['type'].load(path=conf['path'],
                                    name=conf['name'],
                                    setting_name=conf['setting_name'])
        return dataset

    if 'name' in conf:
        dataset = conf['type'].load(path=conf['path'], name=conf['name'])
        return dataset

    try:
        dataset = conf['type'].load(path=conf['path'])
    except Exception as ex:
        print(ex)
        dataset = conf['type'].load(**conf)

    return dataset


def clean_string(value):
    """Helper function to clean and normalize string data.

    It strips leading and trailing whitespace and replaces multiple whitespace
    characters with a single space.
    """
    if isinstance(value, str):
        return ' '.join(value.split())
    return value


class TestingOmDatasets(unittest.TestCase):

    def test_datasets(self):
        # 加载 OpenMind 和 Local 数据集配置
        om_datasets_conf = load_datasets_conf('OpenMind')
        local_datasets_conf = load_datasets_conf('Local')

        # 初始化成功和失败的数据集列表
        successful_comparisons = []
        failed_comparisons = []

        def compare_datasets(om_conf, local_conf):
            openmind_path_name = f"{om_conf.get('path')}/{om_conf.get('name', '')}\t{om_conf.get('lang', '')}"
            local_path_name = f"{local_conf.get('path')}/{local_conf.get('name', '')}\t{local_conf.get('lang', '')}"
            # 断言类型一致
            assert om_conf['type'] == local_conf['type'], "Data types do not match"
            print(openmind_path_name, local_path_name)
            try:
                om_dataset = load_datasets('OpenMind', om_conf)
                local_dataset = load_datasets('Local', local_conf)
                _check_data(om_dataset, local_dataset, sample_size=sample_size)
                return 'success', f'{openmind_path_name} | {local_path_name}'
            except Exception as exception:
                print(exception)
                return 'failure', f'{openmind_path_name} is not the same as {local_path_name}'

        with ThreadPoolExecutor(thread) as executor:
            futures = {
                executor.submit(compare_datasets, om_conf, local_conf): (om_conf, local_conf)
                for om_conf, local_conf in zip(om_datasets_conf, local_datasets_conf)
            }

            for future in tqdm(as_completed(futures), total=len(futures)):
                result, message = future.result()
                if result == 'success':
                    successful_comparisons.append(message)
                else:
                    failed_comparisons.append(message)

        # 输出测试总结
        total_datasets = len(om_datasets_conf)
        print(f"All {total_datasets} datasets")
        print(f"OK {len(successful_comparisons)} datasets")
        for success in successful_comparisons:
            print(f"  {success}")
        print(f"Fail {len(failed_comparisons)} datasets")
        for failure in failed_comparisons:
            print(f"  {failure}")


def _check_data(om_dataset: Dataset | DatasetDict,
                oc_dataset: Dataset | DatasetDict,
                sample_size):
    assert type(om_dataset) == type(
        oc_dataset
    ), f'Dataset type not match: {type(om_dataset)} != {type(oc_dataset)}'

    # match DatasetDict
    if isinstance(oc_dataset, DatasetDict):
        assert om_dataset.keys() == oc_dataset.keys(
        ), f'DatasetDict not match: {om_dataset.keys()} != {oc_dataset.keys()}'

        for key in om_dataset.keys():
            _check_data(om_dataset[key], oc_dataset[key], sample_size=sample_size)

    elif isinstance(oc_dataset, Dataset):
        # match by cols
        assert set(om_dataset.column_names) == set(
            oc_dataset.column_names
        ), f'Column names do not match: {om_dataset.column_names} != {oc_dataset.column_names}'

        # Check that the number of rows is the same
        assert len(om_dataset) == len(
            oc_dataset
        ), f'Number of rows do not match: {len(om_dataset)} != {len(oc_dataset)}'

        # Randomly sample indices
        sample_indices = random.sample(range(len(om_dataset)),
                                       min(sample_size, len(om_dataset)))

        for i, idx in enumerate(sample_indices):
            for col in om_dataset.column_names:
                om_value = clean_string(str(om_dataset[col][idx]))
                oc_value = clean_string(str(oc_dataset[col][idx]))
                try:
                    assert om_value == oc_value, f"Value mismatch in column '{col}', index {idx}: {om_value} != {oc_value}"
                except AssertionError as e:
                    print(f"Assertion failed for column '{col}', index {idx}")
                    print(f"om_data: {om_dataset[idx]}")
                    print(f'oc_data: {oc_dataset[idx]}')
                    print(f'om_value: {om_value} ({type(om_value)})')
                    print(f'oc_value: {oc_value} ({type(oc_value)})')
                    raise e
    else:
        raise ValueError(f'Datasets type not supported {type(om_dataset)}')


if __name__ == '__main__':
    sample_size = 100
    thread = 1
    unittest.main()
