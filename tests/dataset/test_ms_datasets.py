import unittest
from mmengine.config import read_base
from os import environ
import importlib
import sys
from datasets import Dataset, DatasetDict
import random

def reload_datasets():
    modules_to_remove = [
        module_name for module_name in sys.modules
        if module_name.startswith("configs.datasets")
    ]

    for module_name in modules_to_remove:
        del sys.modules[module_name]

    with read_base():
        # from configs.datasets.ceval.ceval_gen import ceval_datasets
        # from configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
        # from configs.datasets.cmmlu.cmmlu_gen import cmmlu_datasets
        # from configs.datasets.ARC_c.ARC_c_gen import ARC_c_datasets
        # from configs.datasets.ARC_c.ARC_c_clean_ppl import ARC_c_datasets as ARC_c_clean_datasets
        # from configs.datasets.ARC_e.ARC_e_gen import ARC_e_datasets
        # from configs.datasets.humaneval.humaneval_gen import humaneval_datasets
        # from configs.datasets.humaneval.humaneval_repeat10_gen_8e312c import humaneval_datasets as humaneval_repeat10_datasets
        # from configs.datasets.race.race_ppl import race_datasets
        # from configs.datasets.commonsenseqa.commonsenseqa_gen import commonsenseqa_datasets
        # from configs.datasets.mmlu.mmlu_gen import mmlu_datasets
        from configs.datasets.mmlu.mmlu_clean_ppl import mmlu_datasets as mmlu_clean_datasets
        

    return sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

def load_datasets_conf(soure):
    environ["DATASET_SOURCE"] = soure
    datasets_conf = reload_datasets()
    return datasets_conf

def load_datasets(source, conf):
    environ["DATASET_SOURCE"] = source
    if 'name' in conf:
        dataset = conf['type'].load(path=conf['path'], name=conf['name'])
    else:
        dataset = conf['type'].load(path=conf['path'])
    return dataset

def clean_string(value):
    """
    Helper function to clean and normalize string data.
    It strips leading and trailing whitespace and replaces multiple whitespace characters with a single space.
    """
    if isinstance(value, str):
        return ' '.join(value.split())
    return value

class TestingMsDatasets(unittest.TestCase):
    def test_datasets(self):
        ms_datasets_conf = load_datasets_conf("ModelScope")
        local_datasets_conf = load_datasets_conf("Local")
        
        for ms_conf, local_conf in zip(ms_datasets_conf, local_datasets_conf):
            print(f"ms file: {ms_conf.get('path')}/{ms_conf.get('name', '')}" )
            print(f"local file: {local_conf.get('path')}/{local_conf.get('name', '')}" )
            
            assert ms_conf['type'] == local_conf['type']
            
            ms_dataset = load_datasets("ModelScope", ms_conf)
            oc_dataset = load_datasets("Local", local_conf)

            _check_data(ms_dataset, oc_dataset)
        
        print(f"All {len(ms_datasets_conf)} datasets are the same!!!" + "="*50)

def _check_data(ms_dataset: Dataset | DatasetDict, oc_dataset: Dataset | DatasetDict, sample_size=100):
    assert type(ms_dataset) == type(oc_dataset), f"Dataset type not match: {type(ms_dataset)} != {type(oc_dataset)}"
  
    # match DatasetDict
    if isinstance(oc_dataset, DatasetDict):
        assert ms_dataset.keys() == oc_dataset.keys(), f"DatasetDict not match: {ms_dataset.keys()} != {oc_dataset.keys()}"
        
        for key in ms_dataset.keys():
            _check_data(ms_dataset[key], oc_dataset[key])
            
    elif isinstance(oc_dataset, Dataset):
        # match by cols
        assert set(ms_dataset.column_names) == set(oc_dataset.column_names), f"Column names do not match: {ms_dataset.column_names} != {oc_dataset.column_names}"

        # Check that the number of rows is the same
        assert len(ms_dataset) == len(oc_dataset), f"Number of rows do not match: {len(ms_dataset)} != {len(oc_dataset)}"

        # Randomly sample indices
        sample_indices = random.sample(range(len(ms_dataset)), min(sample_size, len(ms_dataset)))
        
        for i, idx in enumerate(sample_indices):
            if i == 0:
                print(f"MS Dataset: {ms_dataset[idx]}")
                print(f"OC Dataset: {oc_dataset[idx]}")
            for col in ms_dataset.column_names:
                ms_value = clean_string(str(ms_dataset[col][idx]))
                oc_value = clean_string(str(oc_dataset[col][idx]))
                try:
                    assert ms_value == oc_value, f"Value mismatch in column '{col}', index {idx}: {ms_value} != {oc_value}"
                except AssertionError as e:
                    print(f"Assertion failed for column '{col}', index {idx}")
                    print(f"ms_value: {ms_value} ({type(ms_value)})")
                    print(f"oc_value: {oc_value} ({type(oc_value)})")
                    raise e
        print("All values match!".upper() + "-"*50)
    else:
        raise ValueError(f"Datasets type not supported {type(ms_dataset)}")
        

if __name__ == '__main__':
    unittest.main()