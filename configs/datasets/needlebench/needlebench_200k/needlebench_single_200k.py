from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.needlebench.origin import NeedleBenchOriginDataset
from opencompass.datasets.needlebench.origin import NeedleBenchOriginEvaluator
from opencompass.datasets.needlebench.origin import needlebench_postprocess
from opencompass.datasets.needlebench.origin import needlebench_dataset_postprocess
import math


def logistic(x, L=100, x0=50, k=0.1):
    return round(L / (1 + math.exp(-k * (x - x0))), 3)


def generate_linear_space(start, end, num):
    if num == 1:
        return [start]
    elif num < 1:
        raise ValueError('num must be at least 1.')
    step = (end - start) / (num - 1)
    return [start + step * i for i in range(num)]


def generate_depth_percents(intervals, interval_type):
    if interval_type == 'linear':
        return generate_linear_space(0, 100, intervals)
    elif interval_type == 'sigmoid':
        linear_space = generate_linear_space(0, 100, intervals)
        return [logistic(x) for x in linear_space]
    else:
        raise ValueError('Unsupported interval type')


needlebench_reader_cfg = dict(input_columns=['prompt'], output_column='answer')

needlebench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{prompt}'),
                dict(role='BOT', prompt='{answer}\n'),
            ]
        )
        ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

needlebench_eval_cfg = dict(
    evaluator=dict(type=NeedleBenchOriginEvaluator),
    pred_postprocessor=dict(type=needlebench_postprocess),
    dataset_postprocessor=dict(type=needlebench_dataset_postprocess),
    pred_role='BOT')

# context_lengths = list([16000, 32000, 48000, 64000, 80000, 96000, 112000, 128000, 144000, 160000, 176000, 192000, 200000])
context_lengths = [16000, 48000, 80000, 112000, 128000, 144000, 176000, 200000]
depths_list = [0, 10, 21, 31, 42, 52, 63, 73, 84, 94, 100]

base_path = './data/needlebench'
file_list = ['PaulGrahamEssays.jsonl']
needlebench_en_datasets = []
needle_file_name = 'needles.jsonl'

for original_context_length in context_lengths:
    for depth_percent in depths_list:
        dataset_dict = {
            'abbr': f'Length{original_context_length}'
                    f'Depth{int(depth_percent)}_origin_en_200k',
            'type': NeedleBenchOriginDataset,
            'path': base_path,
            'length': original_context_length,
            'depth': int(depth_percent),
            'tokenizer_model': 'gpt-4',
            'file_list': file_list,
            'num_repeats_per_file': 10,
            'length_buffer': 600,
            'guide': True,
            'language': 'English',
            'needle_file_name': needle_file_name,
            'reader_cfg': needlebench_reader_cfg,
            'infer_cfg': needlebench_infer_cfg,
            'eval_cfg': needlebench_eval_cfg
        }
        needlebench_en_datasets.append(dataset_dict)

file_list = ['zh_finance.jsonl']
needlebench_zh_datasets = []
needle_file_name = 'needles.jsonl'

for original_context_length in context_lengths:
    for depth_percent in depths_list:
        dataset_dict = {
            'abbr': f'Length{original_context_length}'
                    f'Depth{int(depth_percent)}_origin_zh_200k',
            'type': NeedleBenchOriginDataset,
            'path': base_path,
            'length': original_context_length,
            'depth': int(depth_percent),
            'tokenizer_model': 'gpt-4',
            'file_list': file_list,
            'num_repeats_per_file': 10,
            'length_buffer': 200,
            'guide': True,
            'language': 'Chinese',
            'needle_file_name': needle_file_name,
            'reader_cfg': needlebench_reader_cfg,
            'infer_cfg': needlebench_infer_cfg,
            'eval_cfg': needlebench_eval_cfg
        }
        needlebench_zh_datasets.append(dataset_dict)
