import csv
import os

import pytest
import yaml

output_path = 'regression_result_daily'

model_list = ['baichuan2-7b-chat-hf', 'internlm2-chat-1.8b-turbomind', 'internlm2-chat-1.8b-sft-turbomind', 
              'internlm2-chat-7b-turbomind', 'internlm2-chat-7b-sft-turbomind', 'llama-3-8b-instruct-turbomind', 
              'minicpm-2b-dpo-fp32-hf', 'minicpm-2b-sft-bf16-hf', 'minicpm-2b-sft-fp32-hf', 
              'phi-3-mini-4k-instruct-hf', 'qwen2-1.5b-instruct-turbomind', 'qwen2-7b-instruct-turbomind']
dataset_list = [
    'gsm8k', 'race-middle', 'race-high'
]


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
class TestChat:
    """Test cases for chat model."""

    @pytest.mark.parametrize('model, dataset', [(p1, p2) for p1 in model_list
                                                for p2 in dataset_list])
    def test_model_dataset_score(self, baseline_scores, result_scores, model,
                                 dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(result_score, base_score)


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
    if len(csv_files) > 1:
        raise 'have more than 1 result file, please check the result manually'
    if len(csv_files) == 0:
        return None
    return csv_files[0]


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
