from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.needlebench.parallel import NeedleBenchParallelDataset
from opencompass.datasets.needlebench.parallel import NeedleBenchParallelEvaluator
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
    evaluator=dict(type=NeedleBenchParallelEvaluator),
    pred_postprocessor=dict(type=needlebench_postprocess),
    dataset_postprocessor=dict(type=needlebench_dataset_postprocess),
    pred_role='BOT')

context_lengths = list(range(1000, 5000, 1000))
document_depth_percent_intervals = 20
document_depth_percent_interval_type = 'linear'

base_path = './data/needlebench'
file_list = ['PaulGrahamEssays.jsonl']
needlebench_en_datasets = []
needle_file_name = 'needles.jsonl'
depths_float = generate_depth_percents(
            document_depth_percent_intervals,
            document_depth_percent_interval_type)
depths = [int(depth) for depth in depths_float]

for original_context_length in context_lengths:
    dataset_dict = {
        'abbr': f'Length{original_context_length}'
                f'_parallel_en_4k',
        'type': NeedleBenchParallelDataset,
        'path': base_path,
        'needle_file_name': needle_file_name,
        'length': original_context_length,
        'depths': depths,
        'tokenizer_model': 'gpt-4',
        'file_list': file_list,
        'num_repeats_per_file': 25,
        'length_buffer': 1000,
        'guide': True,
        'language': 'English',
        'reader_cfg': needlebench_reader_cfg,
        'infer_cfg': needlebench_infer_cfg,
        'eval_cfg': needlebench_eval_cfg
    }
    needlebench_en_datasets.append(dataset_dict)

file_list = ['zh_finance.jsonl']
needlebench_zh_datasets = []

for original_context_length in context_lengths:
    dataset_dict = {
        'abbr': f'Length{original_context_length}'
                f'_parallel_zh_4k',
        'type': NeedleBenchParallelDataset,
        'path': base_path,
        'needle_file_name': needle_file_name,
        'length': original_context_length,
        'depths': depths,
        'tokenizer_model': 'gpt-4',
        'file_list': file_list,
        'num_repeats_per_file': 25,
        'length_buffer': 200,
        'guide': True,
        'language': 'Chinese',
        'reader_cfg': needlebench_reader_cfg,
        'infer_cfg': needlebench_infer_cfg,
        'eval_cfg': needlebench_eval_cfg
    }
    needlebench_zh_datasets.append(dataset_dict)
