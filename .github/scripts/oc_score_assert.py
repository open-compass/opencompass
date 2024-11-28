import csv
import os

import pytest
import yaml

output_path = 'regression_result_daily'

chat_model_list = [
    'baichuan2-7b-chat-hf', 'glm-4-9b-chat-hf', 'glm-4-9b-chat-turbomind',
    'glm-4-9b-chat-vllm', 'deepseek-7b-chat-hf', 'deepseek-moe-16b-chat-hf',
    'deepseek-7b-chat-vllm', 'gemma2-2b-it-hf', 'gemma2-9b-it-hf',
    'gemma-2b-it-hf', 'gemma-7b-it-hf', 'gemma-2-9b-it-turbomind',
    'gemma-7b-it-vllm', 'internlm2_5-7b-chat-hf',
    'internlm2_5-7b-chat-turbomind', 'internlm2-chat-1.8b-turbomind',
    'internlm2-chat-1.8b-sft-turbomind', 'internlm2-chat-7b-lmdeploy',
    'internlm2-chat-7b-sft-turbomind', 'internlm2-chat-7b-vllm',
    'llama-3_1-8b-instruct-hf', 'llama-3_2-3b-instruct-hf',
    'llama-3-8b-instruct-hf', 'llama-3_1-8b-instruct-turbomind',
    'llama-3_2-3b-instruct-turbomind', 'llama-3-8b-instruct-turbomind',
    'mistral-7b-instruct-v0.2-hf', 'mistral-7b-instruct-v0.3-hf',
    'mistral-nemo-instruct-2407-hf', 'mistral-nemo-instruct-2407-turbomind',
    'mistral-7b-instruct-v0.1-vllm', 'mistral-7b-instruct-v0.2-vllm',
    'MiniCPM3-4B-hf', 'minicpm-2b-dpo-fp32-hf', 'minicpm-2b-sft-bf16-hf',
    'minicpm-2b-sft-fp32-hf', 'phi-3-mini-4k-instruct-hf',
    'qwen1.5-0.5b-chat-hf', 'qwen2-1.5b-instruct-hf', 'qwen2-7b-instruct-hf',
    'qwen2-1.5b-instruct-turbomind', 'qwen2-7b-instruct-turbomind',
    'qwen1.5-0.5b-chat-vllm', 'yi-1.5-6b-chat-hf', 'yi-1.5-9b-chat-hf',
    'deepseek-v2-lite-chat-hf', 'internlm2_5-20b-chat-hf',
    'internlm2_5-20b-chat-turbomind', 'mistral-small-instruct-2409-hf',
    'mistral-small-instruct-2409-turbomind', 'qwen2.5-14b-instruct-hf',
    'qwen2.5-14b-instruct-turbomind'
]
base_model_list = [
    'glm-4-9b-hf', 'deepseek-moe-16b-base-hf', 'deepseek-7b-base-turbomind',
    'deepseek-moe-16b-base-vllm', 'gemma2-2b-hf', 'gemma2-9b-hf',
    'gemma-2b-hf', 'gemma-7b-hf', 'gemma-2b-vllm', 'gemma-7b-vllm',
    'internlm2_5-7b-hf', 'internlm2-7b-hf', 'internlm2-base-7b-hf',
    'internlm2-1.8b-turbomind', 'internlm2_5-7b-turbomind',
    'internlm2-7b-turbomind', 'internlm2-base-7b-turbomind', 'llama-2-7b-hf',
    'llama-3_1-8b-hf', 'llama-3-8b-hf', 'llama-3.1-8b-turbomind',
    'llama-3-8b-turbomind', 'mistral-7b-v0.2-hf', 'mistral-7b-v0.3-hf',
    'mistral-7b-v0.2-vllm', 'qwen2.5-7b-hf', 'qwen2.5-1.5b-turbomind',
    'qwen2.5-7b-turbomind', 'qwen1.5-moe-a2.7b-hf', 'qwen2-0.5b-hf',
    'qwen2-1.5b-hf', 'qwen2-7b-hf', 'qwen2-1.5b-turbomind',
    'qwen2-7b-turbomind', 'qwen1.5-0.5b-vllm', 'yi-1.5-6b-hf', 'yi-1.5-9b-hf',
    'deepseek-v2-lite-hf', 'internlm2-20b-hf', 'internlm2-base-20b-hf',
    'internlm2-20b-turbomind', 'qwen2.5-14b-hf'
]


@pytest.fixture()
def baseline_scores_testrange(request):
    config_path = os.path.join(
        request.config.rootdir,
        '.github/scripts/oc_score_baseline_testrange.yaml')
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return config


@pytest.fixture()
def baseline_scores(request):
    config_path = os.path.join(request.config.rootdir,
                               '.github/scripts/oc_score_baseline.yaml')
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return config


@pytest.fixture()
def baseline_scores_fullbench(request):
    config_path = os.path.join(
        request.config.rootdir,
        '.github/scripts/oc_score_baseline_fullbench.yaml')
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return config


@pytest.fixture()
def result_scores():
    file = find_csv_files(output_path)
    if file is None:
        return None
    return read_csv_file(file)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores_testrange')
@pytest.mark.chat
class TestChat:
    """Test cases for chat model."""

    @pytest.mark.parametrize('model, dataset',
                             [(p1, p2) for p1 in chat_model_list
                              for p2 in ['gsm8k', 'race-high']])
    def test_model_dataset_score(self, baseline_scores_testrange,
                                 result_scores, model, dataset):
        base_score = baseline_scores_testrange.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores_testrange')
@pytest.mark.base
class TestBase:
    """Test cases for base model."""

    @pytest.mark.parametrize(
        'model, dataset',
        [(p1, p2) for p1 in base_model_list
         for p2 in ['gsm8k', 'GPQA_diamond', 'race-high', 'winogrande']])
    def test_model_dataset_score(self, baseline_scores_testrange,
                                 result_scores, model, dataset):
        if model in ['gemma-2b-vllm', 'gemma-7b-vllm'] and dataset != 'gsm8k':
            return
        base_score = baseline_scores_testrange.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores_fullbench')
@pytest.mark.chat_obj_fullbench
class TestChatObjFullbench:
    """Test cases for chat model."""

    @pytest.mark.parametrize('model, dataset', [(p1, p2) for p1 in [
        'internlm2_5-7b-chat-hf_fullbench',
        'internlm2_5-7b-chat-turbomind_fullbench'
    ] for p2 in [
        'race-high', 'ARC-c', 'BoolQ', 'triviaqa_wiki_1shot', 'nq_open_1shot',
        'IFEval', 'drop', 'GPQA_diamond', 'hellaswag', 'TheoremQA',
        'musr_average', 'gsm8k', 'math', 'cmo_fib', 'aime2024',
        'wikibench-wiki-single_choice_cncircular', 'sanitized_mbpp', 'ds1000',
        'lcb_code_generation', 'lcb_code_execution', 'lcb_test_output',
        'bbh-logical_deduction_seven_objects', 'bbh-multistep_arithmetic_two',
        'mmlu-other', 'cmmlu-china-specific', 'mmlu_pro_math', 'ds1000_Pandas',
        'ds1000_Numpy', 'ds1000_Tensorflow', 'ds1000_Scipy', 'ds1000_Sklearn',
        'ds1000_Pytorch', 'ds1000_Matplotlib', 'openai_mmmlu_lite_AR-XY',
        'college', 'college_knowledge'
    ]])
    def test_model_dataset_score(self, baseline_scores_fullbench,
                                 result_scores, model, dataset):
        base_score = baseline_scores_fullbench.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores_fullbench')
@pytest.mark.chat_sub_fullbench
class TestChatSubFullbench:
    """Test cases for chat model."""

    @pytest.mark.parametrize('model, dataset', [(p1, p2) for p1 in [
        'internlm2_5-7b-chat-hf_fullbench',
        'internlm2_5-7b-chat-turbomind_fullbench'
    ] for p2 in [
        'Alignbench总分', 'Alignbench专业能力', 'AlpacaEvaltotal',
        'AlpacaEvalhelpful_base', 'CompassArenacompassarena_language',
        'CompassArenacompassarena_knowledge',
        'CompassArenacompassarena_reason_v2',
        'CompassArenacompassarena_math_v2',
        'CompassArenacompassarena_creationv2_zh', 'Fofofofo_test_prompts',
        'followbenchHSR_AVG', 'followbenchSSR_AVG', 'followbenchHSR_L1',
        'followbenchHSR_L2', 'followbenchHSR_L3', 'followbenchHSR_L4',
        'followbenchHSR_L5', 'followbenchSSR_L1', 'followbenchSSR_L2',
        'followbenchSSR_L3', 'followbenchSSR_L4', 'followbenchSSR_L5',
        'MTBench101average', 'Wildbenchscore'
    ]])
    def test_model_dataset_score(self, baseline_scores_fullbench,
                                 result_scores, model, dataset):
        base_score = baseline_scores_fullbench.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores_fullbench')
@pytest.mark.base_fullbench
class TestBaseFullbench:
    """Test cases for chat model."""

    @pytest.mark.parametrize('model, dataset', [(p1, p2) for p1 in [
        'internlm2_5-7b-hf_fullbench', 'internlm2_5-7b-turbomind_fullbench'
    ] for p2 in [
        'race-high', 'ARC-c', 'BoolQ', 'drop', 'GPQA_diamond', 'math',
        'wikibench-wiki-single_choice_cncircular', 'sanitized_mbpp', 'gsm8k',
        'triviaqa_wiki_1shot', 'nq_open_1shot', 'winogrande', 'hellaswag',
        'TheoremQA', 'dingo_en_192', 'dingo_zh_170', 'college',
        'college_knowledge', 'bbh-logical_deduction_seven_objects',
        'bbh-multistep_arithmetic_two', 'mmlu-other', 'cmmlu-china-specific',
        'mmlu_pro_math'
    ]])
    def test_model_dataset_score(self, baseline_scores_fullbench,
                                 result_scores, model, dataset):
        base_score = baseline_scores_fullbench.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores')
@pytest.mark.api
class TestApibench:
    """Test cases for chat model."""

    @pytest.mark.parametrize('model, dataset',
                             [('lmdeploy-api-test', 'race-middle'),
                              ('lmdeploy-api-test', 'race-high'),
                              ('lmdeploy-api-test', 'gsm8k')])
    def test_api(self, baseline_scores, result_scores, model, dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model + '_batch', result_score, base_score)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores')
class TestCmdCase:

    @pytest.mark.case1
    @pytest.mark.parametrize('model, dataset',
                             [('internlm2_5-7b-hf', 'race-middle'),
                              ('internlm2_5-7b-hf', 'race-high'),
                              ('internlm2_5-7b-hf', 'demo_gsm8k'),
                              ('internlm2-1.8b-hf', 'race-middle'),
                              ('internlm2-1.8b-hf', 'race-high'),
                              ('internlm2-1.8b-hf', 'demo_gsm8k')])
    def test_cmd_case1(self, baseline_scores, result_scores, model, dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)

    @pytest.mark.case2
    @pytest.mark.parametrize('model, dataset',
                             [('internlm2_5-7b-chat-lmdeploy', 'race-middle'),
                              ('internlm2_5-7b-chat-lmdeploy', 'race-high'),
                              ('internlm2_5-7b-chat-lmdeploy', 'demo_gsm8k'),
                              ('internlm2-chat-1.8b-lmdeploy', 'race-middle'),
                              ('internlm2-chat-1.8b-lmdeploy', 'race-high'),
                              ('internlm2-chat-1.8b-lmdeploy', 'demo_gsm8k')])
    def test_cmd_case2(self, baseline_scores, result_scores, model, dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model + '_batch', result_score, base_score)

    @pytest.mark.case3
    @pytest.mark.parametrize('model, dataset',
                             [('internlm2_5-7b_hf', 'race-middle'),
                              ('internlm2_5-7b_hf', 'race-high'),
                              ('internlm2_5-7b_hf', 'demo_gsm8k')])
    def test_cmd_case3(self, baseline_scores, result_scores, model, dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)

    @pytest.mark.case4
    @pytest.mark.parametrize('model, dataset',
                             [('internlm2_5-7b-chat_hf', 'race-middle'),
                              ('internlm2_5-7b-chat_hf', 'race-high'),
                              ('internlm2_5-7b-chat_hf', 'demo_gsm8k')])
    def test_cmd_case4(self, baseline_scores, result_scores, model, dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)


THRESHOLD = 3


def assert_score(model_type, score, baseline):
    if score is None or score == '-':
        assert False, 'value is none'

    if 'batch' not in model_type:
        if float(score) <= (baseline + 0.01) and float(score) >= (baseline -
                                                                  0.01):
            print(' '.join([score, 'is equal', str(baseline)]))
            assert True
        else:
            print(' '.join([score, 'is not equal', str(baseline)]))
            assert False, ' '.join([score, 'is not equal', str(baseline)])
    else:
        if float(score) <= (baseline + THRESHOLD) and float(score) >= (
                baseline - THRESHOLD):
            print(' '.join([
                score, 'is between',
                str(baseline - THRESHOLD), 'and',
                str(baseline + THRESHOLD)
            ]))
            assert True
        else:
            print(' '.join([
                score, 'is not etween',
                str(baseline - THRESHOLD), 'and',
                str(baseline + THRESHOLD)
            ]))
            assert False, ' '.join([
                score, 'is not etween',
                str(baseline - THRESHOLD), 'and',
                str(baseline + THRESHOLD)
            ])


def find_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv') and (file.startswith('summary') or
                                          file.startswith('Subjective_all')):
                csv_files.append(os.path.join(root, file))

    csv_files_with_time = {f: os.path.getctime(f) for f in csv_files}
    sorted_csv_files = sorted(csv_files_with_time.items(), key=lambda x: x[1])
    latest_csv_file = sorted_csv_files[-1][0]
    return latest_csv_file


def read_csv_file(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        filtered_data = []
        if 'Subjective_all' not in file_path:
            for row in reader:
                if row['metric'] is not None and 'bpb' not in row['metric']:
                    filtered_row = {
                        k: v
                        for k, v in row.items()
                        if k not in ['version', 'metric', 'mode']
                    }
                    filtered_data.append(filtered_row)
        else:
            for row in reader:
                if row['Detailed Scores'] is not None:
                    filtered_row = row
                    filtered_row['dataset'] = filtered_row[
                        'Dataset'] + filtered_row['Detailed Scores']
                    del filtered_row['Dataset']
                    del filtered_row['Detailed Scores']
                    filtered_data.append(filtered_row)

    result = {}
    for data in filtered_data:
        dataset = data.get('dataset')
        for key in data.keys():
            if key == 'dataset':
                continue
            else:
                if key in result.keys():
                    result.get(key)[dataset] = data.get(key)
                else:
                    result[key] = {dataset: data.get(key)}
    return result
