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
    ] for p2 in ['race-high', 'ARC-c']])
    def test_model_dataset_score(self, baseline_scores_testrange,
                                 result_scores, model, dataset):
        base_score = baseline_scores_testrange.get(model).get(dataset)
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
    ] for p2 in ['race-high', 'ARC-c']])
    def test_model_dataset_score(self, baseline_scores_testrange,
                                 result_scores, model, dataset):
        base_score = baseline_scores_testrange.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores_fullbench')
@pytest.mark.base_fullbench
class TestBaseFullbench:
    """Test cases for chat model."""

    @pytest.mark.parametrize('model, dataset', [(p1, p2) for p1 in [
        'internlm2_5-7b-chat-hf_fullbench',
        'internlm2_5-7b-chat-turbomind_fullbench'
    ] for p2 in ['race-high', 'ARC-c']])
    def test_model_dataset_score(self, baseline_scores_testrange,
                                 result_scores, model, dataset):
        base_score = baseline_scores_testrange.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)


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
                             [('internlm2_5-7b-hf', 'race-middle'),
                              ('internlm2_5-7b-hf', 'race-high'),
                              ('internlm2_5-7b-hf', 'demo_gsm8k')])
    def test_cmd_case3(self, baseline_scores, result_scores, model, dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)

    @pytest.mark.case4
    @pytest.mark.parametrize('model, dataset',
                             [('internlm2_5-7b-chat-lmdeploy', 'race-middle'),
                              ('internlm2_5-7b-chat-lmdeploy', 'race-high'),
                              ('internlm2_5-7b-chat-lmdeploy', 'demo_gsm8k')])
    def test_cmd_case4(self, baseline_scores, result_scores, model, dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)


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
        if float(score) <= (baseline + 2) and float(score) >= (baseline - 2):
            print(' '.join([
                score, 'is between',
                str(baseline - 2), 'and',
                str(baseline + 2)
            ]))
            assert True
        else:
            print(' '.join([
                score, 'is not etween',
                str(baseline - 2), 'and',
                str(baseline + 2)
            ]))
            assert False, ' '.join([
                score, 'is not etween',
                str(baseline - 2), 'and',
                str(baseline + 2)
            ])


def find_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    csv_files_with_time = {f: os.path.getctime(f) for f in csv_files}
    sorted_csv_files = sorted(csv_files_with_time.items(), key=lambda x: x[1])
    latest_csv_file = sorted_csv_files[-1][0]
    return latest_csv_file


def read_csv_file(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        filtered_data = []

        for row in reader:
            if row['metric'] is not None and 'bpb' not in row['metric']:
                filtered_row = {
                    k: v
                    for k, v in row.items()
                    if k not in ['version', 'metric', 'mode']
                }
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
