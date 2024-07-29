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
        from configs.datasets.ceval.ceval_gen import ceval_datasets
        from configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
        from configs.datasets.cmmlu.cmmlu_gen import cmmlu_datasets
        from configs.datasets.ARC_c.ARC_c_gen import ARC_c_datasets
        from configs.datasets.ARC_e.ARC_e_gen import ARC_e_datasets
        from configs.datasets.humaneval.humaneval_gen import humaneval_datasets
        from configs.datasets.humaneval.humaneval_repeat10_gen_8e312c import humaneval_datasets as humaneval_repeat10_datasets
        from configs.datasets.race.race_ppl import race_datasets
        from configs.datasets.commonsenseqa.commonsenseqa_gen import commonsenseqa_datasets
        
        from configs.datasets.mmlu.mmlu_gen import mmlu_datasets
        from configs.datasets.strategyqa.strategyqa_gen import strategyqa_datasets
        from configs.datasets.bbh.bbh_gen import bbh_datasets
        from configs.datasets.Xsum.Xsum_gen import Xsum_datasets
        from configs.datasets.winogrande.winogrande_gen import winogrande_datasets
        from configs.datasets.winogrande.winogrande_ll import winogrande_datasets as winogrande_ll_datasets
        from configs.datasets.winogrande.winogrande_5shot_ll_252f01 import winogrande_datasets as winogrande_5shot_ll_datasets
        from configs.datasets.obqa.obqa_gen import obqa_datasets
        from configs.datasets.obqa.obqa_ppl_6aac9e import obqa_datasets as obqa_ppl_datasets
        from configs.datasets.agieval.agieval_gen import agieval_datasets as agieval_v2_datasets
        from configs.datasets.agieval.agieval_gen_a0c741 import agieval_datasets as agieval_v1_datasets
        from configs.datasets.siqa.siqa_gen import siqa_datasets as siqa_v2_datasets
        from configs.datasets.siqa.siqa_gen_18632c import siqa_datasets as siqa_v3_datasets
        from configs.datasets.siqa.siqa_ppl_42bc6e import siqa_datasets as siqa_ppl_datasets
        from configs.datasets.storycloze.storycloze_gen import storycloze_datasets
        from configs.datasets.storycloze.storycloze_ppl import storycloze_datasets as storycloze_ppl_datasets
        from configs.datasets.summedits.summedits_gen import summedits_datasets as summedits_v2_datasets
        
        from configs.datasets.hellaswag.hellaswag_gen import hellaswag_datasets as hellaswag_v2_datasets
        from configs.datasets.hellaswag.hellaswag_10shot_gen_e42710 import hellaswag_datasets as hellaswag_ice_datasets
        from configs.datasets.hellaswag.hellaswag_ppl_9dbb12 import hellaswag_datasets as hellaswag_v1_datasets
        from configs.datasets.hellaswag.hellaswag_ppl_a6e128 import hellaswag_datasets as hellaswag_v3_datasets
        from configs.datasets.mbpp.mbpp_gen import mbpp_datasets as mbpp_v1_datasets
        from configs.datasets.mbpp.mbpp_passk_gen_830460 import mbpp_datasets as mbpp_v2_datasets
        from configs.datasets.mbpp.sanitized_mbpp_gen_830460 import sanitized_mbpp_datasets
        from configs.datasets.nq.nq_gen import nq_datasets
        from configs.datasets.lcsts.lcsts_gen import lcsts_datasets
        from configs.datasets.math.math_gen import math_datasets
        from configs.datasets.piqa.piqa_gen import piqa_datasets as piqa_v2_datasets
        from configs.datasets.piqa.piqa_ppl import piqa_datasets as piqa_v1_datasets
        from configs.datasets.piqa.piqa_ppl_0cfff2 import piqa_datasets as piqa_v3_datasets
        from configs.datasets.lambada.lambada_gen import lambada_datasets
        from configs.datasets.tydiqa.tydiqa_gen import tydiqa_datasets
        from configs.datasets.GaokaoBench.GaokaoBench_gen import GaokaoBench_datasets
        from configs.datasets.GaokaoBench.GaokaoBench_mixed import GaokaoBench_datasets as GaokaoBench_mixed_datasets
        from configs.datasets.GaokaoBench.GaokaoBench_no_subjective_gen_4c31db import GaokaoBench_datasets as GaokaoBench_no_subjective_datasets
        from configs.datasets.triviaqa.triviaqa_gen import triviaqa_datasets
        from configs.datasets.triviaqa.triviaqa_wiki_1shot_gen_20a989 import triviaqa_datasets as triviaqa_wiki_1shot_datasets
        
        from configs.datasets.CLUE_cmnli.CLUE_cmnli_gen import cmnli_datasets
        from configs.datasets.CLUE_cmnli.CLUE_cmnli_ppl import cmnli_datasets as cmnli_ppl_datasets
        from configs.datasets.CLUE_ocnli.CLUE_ocnli_gen import ocnli_datasets
        
        from configs.datasets.ceval.ceval_clean_ppl import ceval_datasets as ceval_clean_datasets
        from configs.datasets.ARC_c.ARC_c_clean_ppl import ARC_c_datasets as ARC_c_clean_datasets
        from configs.datasets.mmlu.mmlu_clean_ppl import mmlu_datasets as mmlu_clean_datasets
        from configs.datasets.hellaswag.hellaswag_clean_ppl import hellaswag_datasets as hellaswag_clean_datasets
        
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
    except Exception:
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


class TestingMsDatasets(unittest.TestCase):

    def test_datasets(self):
        # 加载 ModelScope 和 Local 数据集配置
        ms_datasets_conf = load_datasets_conf('ModelScope')
        local_datasets_conf = load_datasets_conf('Local')
        
        # 初始化成功和失败的数据集列表
        successful_comparisons = []
        failed_comparisons = []
        
        def compare_datasets(ms_conf, local_conf):
            modelscope_path_name = f"{ms_conf.get('path')}/{ms_conf.get('name', '')}\t{ms_conf.get('lang', '')}"
            local_path_name = f"{local_conf.get('path')}/{local_conf.get('name', '')}\t{local_conf.get('lang', '')}"
            # 断言类型一致
            assert ms_conf['type'] == local_conf['type'], "Data types do not match"
            print(modelscope_path_name, local_path_name)
            try:
                ms_dataset = load_datasets('ModelScope', ms_conf)
                local_dataset = load_datasets('Local', local_conf)
                _check_data(ms_dataset, local_dataset, sample_size=sample_size)
                return 'success', f'{modelscope_path_name} | {local_path_name}'
            except Exception as exception:
                print(exception)
                return 'failure', f'{modelscope_path_name} is not the same as {local_path_name}'
        
        with ThreadPoolExecutor(16) as executor:
            futures = {
                executor.submit(compare_datasets, ms_conf, local_conf): (ms_conf, local_conf)
                for ms_conf, local_conf in zip(ms_datasets_conf, local_datasets_conf)
            }
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                result, message = future.result()
                if result == 'success':
                    successful_comparisons.append(message)
                else:
                    failed_comparisons.append(message)
        
        # 输出测试总结
        total_datasets = len(ms_datasets_conf)
        print(f"All {total_datasets} datasets")
        print(f"OK {len(successful_comparisons)} datasets")
        for success in successful_comparisons:
            print(f"  {success}")
        print(f"Fail {len(failed_comparisons)} datasets")
        for failure in failed_comparisons:
            print(f"  {failure}")


def _check_data(ms_dataset: Dataset | DatasetDict,
                oc_dataset: Dataset | DatasetDict,
                sample_size):
    assert type(ms_dataset) == type(
        oc_dataset
    ), f'Dataset type not match: {type(ms_dataset)} != {type(oc_dataset)}'

    # match DatasetDict
    if isinstance(oc_dataset, DatasetDict):
        assert ms_dataset.keys() == oc_dataset.keys(
        ), f'DatasetDict not match: {ms_dataset.keys()} != {oc_dataset.keys()}'

        for key in ms_dataset.keys():
            _check_data(ms_dataset[key], oc_dataset[key], sample_size=sample_size)

    elif isinstance(oc_dataset, Dataset):
        # match by cols
        assert set(ms_dataset.column_names) == set(
            oc_dataset.column_names
        ), f'Column names do not match: {ms_dataset.column_names} != {oc_dataset.column_names}'

        # Check that the number of rows is the same
        assert len(ms_dataset) == len(
            oc_dataset
        ), f'Number of rows do not match: {len(ms_dataset)} != {len(oc_dataset)}'

        # Randomly sample indices
        sample_indices = random.sample(range(len(ms_dataset)),
                                       min(sample_size, len(ms_dataset)))

        for i, idx in enumerate(sample_indices):
            for col in ms_dataset.column_names:
                ms_value = clean_string(str(ms_dataset[col][idx]))
                oc_value = clean_string(str(oc_dataset[col][idx]))
                try:
                    assert ms_value == oc_value, f"Value mismatch in column '{col}', index {idx}: {ms_value} != {oc_value}"
                except AssertionError as e:
                    print(f"Assertion failed for column '{col}', index {idx}")
                    print(f"ms_data: {ms_dataset[idx]}")
                    print(f'oc_data: {oc_dataset[idx]}')
                    print(f'ms_value: {ms_value} ({type(ms_value)})')
                    print(f'oc_value: {oc_value} ({type(oc_value)})')
                    raise e
    else:
        raise ValueError(f'Datasets type not supported {type(ms_dataset)}')


if __name__ == '__main__':
    sample_size = 100
    unittest.main()
