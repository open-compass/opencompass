import os

DATASETS_MAPPING = {
    # ADVGLUE Datasets
    'opencompass/advglue-dev': {
        'ms_id': None,
        'hf_id': None,
        'local': './data/adv_glue/dev_ann.json',
    },
    # AGIEval Datasets
    'opencompass/agieval': {
        'ms_id': 'opencompass/agieval',
        'hf_id': 'opencompass/agieval',
        'local': './data/AGIEval/data/v1/',
    },

    # ARC Datasets(Test)
    'opencompass/ai2_arc-test': {
        'ms_id': 'opencompass/ai2_arc',
        'hf_id': 'opencompass/ai2_arc',
        'local': './data/ARC/ARC-c/ARC-Challenge-Test.jsonl',
    },
    'opencompass/ai2_arc-dev': {
        'ms_id': 'opencompass/ai2_arc',
        'hf_id': 'opencompass/ai2_arc',
        'local': './data/ARC/ARC-c/ARC-Challenge-Dev.jsonl',
    },
    'opencompass/ai2_arc-easy-dev': {
        'ms_id': 'opencompass/ai2_arc',
        'hf_id': 'opencompass/ai2_arc',
        'local': './data/ARC/ARC-e/ARC-Easy-Dev.jsonl',
    },
    # BBH
    'opencompass/bbh': {
        'ms_id': 'opencompass/bbh',
        'hf_id': 'opencompass/bbh',
        'local': './data/BBH/data',
    },
    # C-Eval
    'opencompass/ceval-exam': {
        'ms_id': 'opencompass/ceval-exam',
        'hf_id': 'opencompass/ceval-exam',
        'local': './data/ceval/formal_ceval',
    },
    # AFQMC
    'opencompass/afqmc-dev': {
        'ms_id': 'opencompass/afqmc',
        'hf_id': 'opencompass/afqmc',
        'local': './data/CLUE/AFQMC/dev.json',
    },
    # CMNLI
    'opencompass/cmnli-dev': {
        'ms_id': 'opencompass/cmnli',
        'hf_id': 'opencompass/cmnli',
        'local': './data/CLUE/cmnli/cmnli_public/dev.json',
    },
    # OCNLI
    'opencompass/OCNLI-dev': {
        'ms_id': 'opencompass/OCNLI',
        'hf_id': 'opencompass/OCNLI',
        'local': './data/CLUE/OCNLI/dev.json',
    },
    # ChemBench
    'opencompass/ChemBench': {
        'ms_id': 'opencompass/ChemBench',
        'hf_id': 'opencompass/ChemBench',
        'local': './data/ChemBench/',
    },
    # CMMLU
    'opencompass/cmmlu': {
        'ms_id': 'opencompass/cmmlu',
        'hf_id': 'opencompass/cmmlu',
        'local': './data/cmmlu/',
    },
    # CommonsenseQA
    'opencompass/commonsense_qa': {
        'ms_id': 'opencompass/commonsense_qa',
        'hf_id': 'opencompass/commonsense_qa',
        'local': './data/commonsenseqa',
    },
    # CMRC
    'opencompass/cmrc_dev': {
        'ms_id': 'opencompass/cmrc_dev',
        'hf_id': 'opencompass/cmrc_dev',
        'local': './data/CLUE/CMRC/dev.json'
    },
    # DRCD_dev
    'opencompass/drcd_dev': {
        'ms_id': 'opencompass/drcd_dev',
        'hf_id': 'opencompass/drcd_dev',
        'local': './data/CLUE/DRCD/dev.json'
    },
    # clozeTest_maxmin
    'opencompass/clozeTest_maxmin': {
        'ms_id': None,
        'hf_id': None,
        'local': './data/clozeTest-maxmin/python/clozeTest.json',
    },
    # clozeTest_maxmin
    'opencompass/clozeTest_maxmin_answers': {
        'ms_id': None,
        'hf_id': None,
        'local': './data/clozeTest-maxmin/python/answers.txt',
    },
    # Flores
    'opencompass/flores': {
        'ms_id': 'opencompass/flores',
        'hf_id': 'opencompass/flores',
        'local': './data/flores_first100',
    },
    # MBPP
    'opencompass/mbpp': {
        'ms_id': 'opencompass/mbpp',
        'hf_id': 'opencompass/mbpp',
        'local': './data/mbpp/mbpp.jsonl',
    },
    # 'opencompass/mbpp': {
    #     'ms_id': 'opencompass/mbpp',
    #     'hf_id': 'opencompass/mbpp',
    #     'local': './data/mbpp/mbpp.jsonl',
    # },
    'opencompass/sanitized_mbpp': {
        'ms_id': 'opencompass/mbpp',
        'hf_id': 'opencompass/mbpp',
        'local': './data/mbpp/sanitized-mbpp.jsonl',
    },
    # GSM
    'opencompass/gsm8k': {
        'ms_id': 'opencompass/gsm8k',
        'hf_id': 'opencompass/gsm8k',
        'local': './data/gsm8k/',
    },
    # HellaSwag
    'opencompass/hellaswag': {
        'ms_id': 'opencompass/hellaswag',
        'hf_id': 'opencompass/hellaswag',
        'local': './data/hellaswag/hellaswag.jsonl',
    },
    # HellaSwagICE
    'opencompass/hellaswag_ice': {
        'ms_id': 'opencompass/hellaswag',
        'hf_id': 'opencompass/hellaswag',
        'local': './data/hellaswag/',
    },
    # HumanEval
    'opencompass/humaneval': {
        'ms_id': 'opencompass/humaneval',
        'hf_id': 'opencompass/humaneval',
        'local': './data/humaneval/human-eval-v2-20210705.jsonl',
    },
    # HumanEvalCN
    'opencompass/humaneval_cn': {
        'ms_id': 'opencompass/humaneval',
        'hf_id': 'opencompass/humaneval',
        'local': './data/humaneval_cn/human-eval-cn-v2-20210705.jsonl',
    },
    # Lambada
    'opencompass/lambada': {
        'ms_id': 'opencompass/lambada',
        'hf_id': 'opencompass/lambada',
        'local': './data/lambada/test.jsonl',
    },
    # LCSTS
    'opencompass/LCSTS': {
        'ms_id': 'opencompass/LCSTS',
        'hf_id': 'opencompass/LCSTS',
        'local': './data/LCSTS',
    },
    # MATH
    'opencompass/math': {
        'ms_id': 'opencompass/math',
        'hf_id': 'opencompass/math',
        'local': './data/math/math.json',
    },
    # MMLU
    'opencompass/mmlu': {
        'ms_id': 'opencompass/mmlu',
        'hf_id': 'opencompass/mmlu',
        'local': './data/mmlu/',
    },
    # NQ
    'opencompass/natural_question': {
        'ms_id': 'opencompass/natural_question',
        'hf_id': 'opencompass/natural_question',
        'local': './data/nq/',
    },
    # OpenBook QA-test
    'opencompass/openbookqa_test': {
        'ms_id': 'opencompass/openbookqa',
        'hf_id': 'opencompass/openbookqa',
        'local': './data/openbookqa/Main/test.jsonl',
    },
    # OpenBook QA-fact
    'opencompass/openbookqa_fact': {
        'ms_id': 'opencompass/openbookqa',
        'hf_id': 'opencompass/openbookqa',
        'local': './data/openbookqa/Additional/test_complete.jsonl',
    },
    # PIQA
    'opencompass/piqa': {
        'ms_id': 'opencompass/piqa',
        'hf_id': 'opencompass/piqa',
        'local': './data/piqa',
    },
    # RACE
    'opencompass/race': {
        'ms_id': 'opencompass/race',
        'hf_id': 'opencompass/race',
        'local': './data/race',
    },
    # SIQA
    'opencompass/siqa': {
        'ms_id': 'opencompass/siqa',
        'hf_id': 'opencompass/siqa',
        'local': './data/siqa',
    },
    # XStoryCloze
    'opencompass/xstory_cloze': {
        'ms_id': 'opencompass/xstory_cloze',
        'hf_id': 'opencompass/xstory_cloze',
        'local': './data/xstory_cloze',
    },
    # StrategyQA
    'opencompass/strategy_qa': {
        'ms_id': 'opencompass/strategy_qa',
        'hf_id': 'opencompass/strategy_qa',
        'local': './data/strategyqa/strategyQA_train.json',
    },
    # SummEdits
    'opencompass/summedits': {
        'ms_id': 'opencompass/summedits',
        'hf_id': 'opencompass/summedits',
        'local': './data/summedits/summedits.jsonl',
    },
    # TriviaQA
    'opencompass/trivia_qa': {
        'ms_id': 'opencompass/trivia_qa',
        'hf_id': 'opencompass/trivia_qa',
        'local': './data/triviaqa/',
    },
    # TydiQA
    'opencompass/tydiqa': {
        'ms_id': 'opencompass/tydiqa',
        'hf_id': 'opencompass/tydiqa',
        'local': './data/tydiqa/',
    },
    # Winogrande
    'opencompass/winogrande': {
        'ms_id': 'opencompass/winogrande',
        'hf_id': 'opencompass/winogrande',
        'local': './data/winogrande/',
    },
    # XSum
    'opencompass/xsum': {
        'ms_id': 'opencompass/xsum',
        'hf_id': 'opencompass/xsum',
        'local': './data/Xsum/dev.jsonl',
    }
}


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
