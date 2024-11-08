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
                # dict(role='BOT', prompt='{answer}\n'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

needlebench_eval_cfg = dict(
    evaluator=dict(type=NeedleBenchOriginEvaluator),
    pred_postprocessor=dict(type=needlebench_postprocess),
    dataset_postprocessor=dict(type=needlebench_dataset_postprocess),
    pred_role='BOT',
)

context_lengths = [32000, 100000, 200000, ]  # 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000
depths_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, ]  # 

base_path = 'opencompass/needlebench'
file_list = ['en_un_asr.jsonl']
needlebench_en_datasets = []
needle_file_name = 'needles.jsonl'

for original_context_length in context_lengths:
    for depth_percent in depths_list:
        dataset_dict = {
            'abbr': f'Length{original_context_length}'
            f'Depth{int(depth_percent)}_origin_en_{original_context_length}',
            'type': NeedleBenchOriginDataset,
            'path': base_path,
            'length': original_context_length,
            'depth': int(depth_percent),
            'tokenizer_model': 'gpt-4',
            'file_list': file_list,
            'num_repeats_per_file': 10,
            'length_buffer': 600,
            'guide': False,
            'language': 'English',
            'needle_file_name': needle_file_name,
            'reader_cfg': needlebench_reader_cfg,
            'infer_cfg': needlebench_infer_cfg,
            'eval_cfg': needlebench_eval_cfg,
        }
        needlebench_en_datasets.append(dataset_dict)

file_list = ['zh_all.jsonl']
needlebench_zh_datasets = []
needle_file_name = 'needles.jsonl'

for original_context_length in context_lengths:
    for depth_percent in depths_list:
        dataset_dict = {
            'abbr': f'Length{original_context_length}'
            f'Depth{int(depth_percent)}_origin_zh_{original_context_length}',
            'type': NeedleBenchOriginDataset,
            'path': base_path,
            'length': original_context_length,
            'depth': int(depth_percent),
            'tokenizer_model': 'gpt-4',
            'file_list': file_list,
            'num_repeats_per_file': 10,
            'length_buffer': 200,
            'guide': False,
            'language': 'Chinese',
            'needle_file_name': needle_file_name,
            'reader_cfg': needlebench_reader_cfg,
            'infer_cfg': needlebench_infer_cfg,
            'eval_cfg': needlebench_eval_cfg,
        }
        needlebench_zh_datasets.append(dataset_dict)
