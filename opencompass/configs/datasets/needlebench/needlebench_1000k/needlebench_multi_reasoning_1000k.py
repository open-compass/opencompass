from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.needlebench.multi import NeedleBenchMultiDataset
from opencompass.datasets.needlebench.multi import NeedleBenchMultiEvaluator
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
    evaluator=dict(type=NeedleBenchMultiEvaluator),
    pred_postprocessor=dict(type=needlebench_postprocess),
    dataset_postprocessor=dict(type=needlebench_dataset_postprocess),
    pred_role='BOT')

context_lengths = [20000, 160000, 300000, 440000, 580000, 720000, 860000, 1000000]
depths_list = [0, 10, 21, 31, 42, 52, 63, 73, 84, 94, 100]

# ----------English Version----------
base_path = './data/needlebench'
file_list = ['PaulGrahamEssays.jsonl']

needle_file_name = 'multi_needle_reasoning_en.json'
diff = 10
num_needles = 2
needlebench_2needle_en_datasets = []
language = 'English'

for original_context_length in context_lengths:
    for depth_percent in depths_list:
        dataset_dict = {
            'abbr': f'Length{original_context_length}'
                    f'Depth{int(depth_percent)}_{num_needles}needle_en_1000k',
            'type': NeedleBenchMultiDataset,
            'path': base_path,
            'length': original_context_length,
            'depth': int(depth_percent),
            'tokenizer_model': 'gpt-4',
            'file_list': file_list,
            'num_repeats_per_file': 10,
            'length_buffer': 600,
            'guide': True,
            'language': language,
            'needle_file_name': needle_file_name,
            'num_needles': num_needles,
            'diff': diff,
            'reader_cfg': needlebench_reader_cfg,
            'infer_cfg': needlebench_infer_cfg,
            'eval_cfg': needlebench_eval_cfg
        }
        needlebench_2needle_en_datasets.append(dataset_dict)

num_needles = 3
needlebench_3needle_en_datasets = []

for original_context_length in context_lengths:
    for depth_percent in depths_list:
        dataset_dict = {
            'abbr': f'Length{original_context_length}'
                    f'Depth{int(depth_percent)}_{num_needles}needle_en_1000k',
            'type': NeedleBenchMultiDataset,
            'path': base_path,
            'length': original_context_length,
            'depth': int(depth_percent),
            'tokenizer_model': 'gpt-4',
            'file_list': file_list,
            'num_repeats_per_file': 10,
            'length_buffer': 600,
            'guide': True,
            'language': language,
            'needle_file_name': needle_file_name,
            'num_needles': num_needles,
            'diff': diff,
            'reader_cfg': needlebench_reader_cfg,
            'infer_cfg': needlebench_infer_cfg,
            'eval_cfg': needlebench_eval_cfg
        }
        needlebench_3needle_en_datasets.append(dataset_dict)

num_needles = 4
needlebench_4needle_en_datasets = []

for original_context_length in context_lengths:
    for depth_percent in depths_list:
        dataset_dict = {
            'abbr': f'Length{original_context_length}'
                    f'Depth{int(depth_percent)}_{num_needles}needle_en_1000k',
            'type': NeedleBenchMultiDataset,
            'path': base_path,
            'length': original_context_length,
            'depth': int(depth_percent),
            'tokenizer_model': 'gpt-4',
            'file_list': file_list,
            'num_repeats_per_file': 10,
            'length_buffer': 600,
            'guide': True,
            'language': language,
            'needle_file_name': needle_file_name,
            'num_needles': num_needles,
            'diff': diff,
            'reader_cfg': needlebench_reader_cfg,
            'infer_cfg': needlebench_infer_cfg,
            'eval_cfg': needlebench_eval_cfg
        }
        needlebench_4needle_en_datasets.append(dataset_dict)

num_needles = 5
needlebench_5needle_en_datasets = []

for original_context_length in context_lengths:
    for depth_percent in depths_list:
        dataset_dict = {
            'abbr': f'Length{original_context_length}'
                    f'Depth{int(depth_percent)}_{num_needles}needle_en_1000k',
            'type': NeedleBenchMultiDataset,
            'path': base_path,
            'length': original_context_length,
            'depth': int(depth_percent),
            'tokenizer_model': 'gpt-4',
            'file_list': file_list,
            'num_repeats_per_file': 10,
            'length_buffer': 600,
            'guide': True,
            'language': language,
            'needle_file_name': needle_file_name,
            'num_needles': num_needles,
            'diff': diff,
            'reader_cfg': needlebench_reader_cfg,
            'infer_cfg': needlebench_infer_cfg,
            'eval_cfg': needlebench_eval_cfg
        }
        needlebench_5needle_en_datasets.append(dataset_dict)

# ----------Chinese Version----------
base_path = './data/needlebench'
file_list = ['zh_finance.jsonl']

needle_file_name = 'multi_needle_reasoning_zh.json'
diff = 10
num_needles = 2
needlebench_2needle_zh_datasets = []
language = 'Chinese'

for original_context_length in context_lengths:
    for depth_percent in depths_list:
        dataset_dict = {
            'abbr': f'Length{original_context_length}'
                    f'Depth{int(depth_percent)}_{num_needles}needle_zh_1000k',
            'type': NeedleBenchMultiDataset,
            'path': base_path,
            'length': original_context_length,
            'depth': int(depth_percent),
            'tokenizer_model': 'gpt-4',
            'file_list': file_list,
            'num_repeats_per_file': 10,
            'length_buffer': 200,
            'guide': True,
            'language': language,
            'needle_file_name': needle_file_name,
            'num_needles': num_needles,
            'diff': diff,
            'reader_cfg': needlebench_reader_cfg,
            'infer_cfg': needlebench_infer_cfg,
            'eval_cfg': needlebench_eval_cfg
        }
        needlebench_2needle_zh_datasets.append(dataset_dict)

num_needles = 3
needlebench_3needle_zh_datasets = []

for original_context_length in context_lengths:
    for depth_percent in depths_list:
        dataset_dict = {
            'abbr': f'Length{original_context_length}'
                    f'Depth{int(depth_percent)}_{num_needles}needle_zh_1000k',
            'type': NeedleBenchMultiDataset,
            'path': base_path,
            'length': original_context_length,
            'depth': int(depth_percent),
            'tokenizer_model': 'gpt-4',
            'file_list': file_list,
            'num_repeats_per_file': 10,
            'length_buffer': 200,
            'guide': True,
            'language': language,
            'needle_file_name': needle_file_name,
            'num_needles': num_needles,
            'diff': diff,
            'reader_cfg': needlebench_reader_cfg,
            'infer_cfg': needlebench_infer_cfg,
            'eval_cfg': needlebench_eval_cfg
        }
        needlebench_3needle_zh_datasets.append(dataset_dict)

num_needles = 4
needlebench_4needle_zh_datasets = []

for original_context_length in context_lengths:
    for depth_percent in depths_list:
        dataset_dict = {
            'abbr': f'Length{original_context_length}'
                    f'Depth{int(depth_percent)}_{num_needles}needle_zh_1000k',
            'type': NeedleBenchMultiDataset,
            'path': base_path,
            'length': original_context_length,
            'depth': int(depth_percent),
            'tokenizer_model': 'gpt-4',
            'file_list': file_list,
            'num_repeats_per_file': 10,
            'length_buffer': 200,
            'guide': True,
            'language': language,
            'needle_file_name': needle_file_name,
            'num_needles': num_needles,
            'diff': diff,
            'reader_cfg': needlebench_reader_cfg,
            'infer_cfg': needlebench_infer_cfg,
            'eval_cfg': needlebench_eval_cfg
        }
        needlebench_4needle_zh_datasets.append(dataset_dict)

num_needles = 5
needlebench_5needle_zh_datasets = []

for original_context_length in context_lengths:
    for depth_percent in depths_list:
        dataset_dict = {
            'abbr': f'Length{original_context_length}'
                    f'Depth{int(depth_percent)}_{num_needles}needle_zh_1000k',
            'type': NeedleBenchMultiDataset,
            'path': base_path,
            'length': original_context_length,
            'depth': int(depth_percent),
            'tokenizer_model': 'gpt-4',
            'file_list': file_list,
            'num_repeats_per_file': 10,
            'length_buffer': 200,
            'guide': True,
            'language': language,
            'needle_file_name': needle_file_name,
            'num_needles': num_needles,
            'diff': diff,
            'reader_cfg': needlebench_reader_cfg,
            'infer_cfg': needlebench_infer_cfg,
            'eval_cfg': needlebench_eval_cfg
        }
        needlebench_5needle_zh_datasets.append(dataset_dict)
