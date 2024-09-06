import csv
import os

import pytest
import yaml

output_path = 'regression_result_daily'

chat_model_list = [
    'baichuan2-7b-chat-hf', 'deepseek-7b-chat-hf', 'deepseek-moe-16b-chat-hf',
    'deepseek-7b-chat-vllm', 'gemma-2b-it-hf', 'gemma-7b-it-hf',
    'internlm2_5-7b-chat-hf', 'internlm2_5-7b-chat-turbomind',
    'internlm2-chat-1.8b-turbomind', 'internlm2-chat-1.8b-sft-turbomind',
    'internlm2-chat-7b-turbomind', 'internlm2-chat-7b-sft-turbomind',
    'internlm2-chat-7b-vllm', 'llama-3-8b-instruct-hf',
    'llama-3-8b-instruct-turbomind', 'mistral-7b-instruct-v0.2-hf',
    'mistral-7b-instruct-v0.2-vllm', 'minicpm-2b-dpo-fp32-hf',
    'minicpm-2b-sft-bf16-hf', 'minicpm-2b-sft-fp32-hf',
    'phi-3-mini-4k-instruct-hf', 'qwen1.5-0.5b-chat-hf',
    'qwen2-1.5b-instruct-turbomind', 'qwen2-7b-instruct-turbomind',
    'qwen1.5-0.5b-chat-vllm', 'yi-1.5-6b-chat-hf', 'yi-1.5-9b-chat-hf',
    'lmdeploy-api-test'
]
base_model_list = [
    'deepseek-moe-16b-base-hf', 'deepseek-7b-base-turbomind',
    'deepseek-moe-16b-base-vllm', 'gemma-2b-hf', 'gemma-7b-hf',
    'internlm2_5-7b-hf', 'internlm2-7b-hf', 'internlm2-base-7b-hf',
    'internlm2_5-7b-turbomind', 'internlm2-1.8b-turbomind',
    'internlm2-7b-turbomind', 'internlm2-base-7b-hf',
    'internlm2-base-7b-turbomind', 'llama-3-8b-turbomind',
    'mistral-7b-v0.2-hf', 'mistral-7b-v0.2-vllm', 'qwen1.5-moe-a2.7b-hf',
    'qwen2-0.5b-hf', 'qwen2-1.5b-turbomind', 'qwen2-7b-turbomind',
    'qwen1.5-0.5b-vllm', 'yi-1.5-6b-hf', 'yi-1.5-9b-hf'
]
dataset_list = ['gsm8k', 'race-middle', 'race-high']


@pytest.fixture()
def baseline_scores(request):
    config_path = os.path.join(request.config.rootdir,
                               '.github/scripts/oc_score_baseline.yaml')
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
@pytest.mark.usefixtures('baseline_scores')
@pytest.mark.chat
class TestChat:
    """Test cases for chat model."""

    @pytest.mark.parametrize('model, dataset', [(p1, p2)
                                                for p1 in chat_model_list
                                                for p2 in dataset_list])
    def test_model_dataset_score(self, baseline_scores, result_scores, model,
                                 dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(result_score, base_score)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores')
@pytest.mark.base
class TestBase:
    """Test cases for base model."""

    @pytest.mark.parametrize('model, dataset', [(p1, p2)
                                                for p1 in base_model_list
                                                for p2 in dataset_list])
    def test_model_dataset_score(self, baseline_scores, result_scores, model,
                                 dataset):
        if model == 'mistral-7b-v0.2-vllm' and dataset == 'race-high':
            return
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(result_score, base_score)


@pytest.mark.usefixtures('result_scores')
class TestCmdCase:

    @pytest.mark.case1
    @pytest.mark.parametrize('model, dataset',
                             [('internlm2_5-7b-hf', 'race-middle'),
                              ('internlm2_5-7b-hf', 'race-high')])
    def test_cmd_case1(self, result_scores, model, dataset):
        if len(result_scores.keys()) != 1:
            assert False, 'result is none'
        result_score = result_scores.get(model).get(dataset)
        assert_score(result_score, 91)

    @pytest.mark.case2
    @pytest.mark.parametrize('model, dataset',
                             [('internlm2_5-7b-chat-turbomind', 'race-middle'),
                              ('internlm2_5-7b-chat-turbomind', 'race-high')])
    def test_cmd_case2(self, result_scores, model, dataset):
        if len(result_scores.keys()) != 1:
            assert False, 'result is none'
        result_score = result_scores.get(model).get(dataset)
        assert_score(result_score, 91)

    @pytest.mark.case3
    @pytest.mark.parametrize('model, dataset',
                             [('internlm2_5-7b_hf', 'race-middle'),
                              ('internlm2_5-7b_hf', 'race-high')])
    def test_cmd_case3(self, result_scores, model, dataset):
        if len(result_scores.keys()) != 1:
            assert False, 'result is none'
        result_score = result_scores.get(model).get(dataset)
        assert_score(result_score, 91)

    @pytest.mark.case4
    @pytest.mark.parametrize('model, dataset',
                             [('internlm2_5-7b-chat_hf', 'race-middle'),
                              ('internlm2_5-7b-chat_hf', 'race-high')])
    def test_cmd_case4(self, result_scores, model, dataset):
        if len(result_scores.keys()) != 1:
            assert False, 'result is none'
        result_score = result_scores.get(model).get(dataset)
        assert_score(result_score, 91)


def assert_score(score, baseline):
    if score is None or score == '-':
        assert False, 'value is none'
    if float(score) <= (baseline + 5) and float(score) >= (baseline - 5):
        print(score + ' between ' + str(baseline - 5) + ' and ' +
              str(baseline + 5))
        assert True
    else:
        assert False, score + ' not between ' + str(
            baseline - 5) + ' and ' + str(baseline + 5)


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
