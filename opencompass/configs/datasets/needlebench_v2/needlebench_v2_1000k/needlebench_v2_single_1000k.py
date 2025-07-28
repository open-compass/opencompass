from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.needlebench_v2.origin import NeedleBenchOriginDataset
from opencompass.datasets.needlebench_v2.origin import NeedleBenchOriginEvaluator
from opencompass.datasets.needlebench_v2.origin import needlebench_postprocess
from opencompass.datasets.needlebench_v2.origin import needlebench_dataset_postprocess


needlebench_reader_cfg = dict(input_columns=['prompt'], output_column='answer')

needlebench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{prompt}'),
                dict(role='BOT', prompt='{answer}\n'),
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

context_lengths = list([1000, 125000, 250000, 375000, 500000, 625000, 750000, 875000, 1000000])
depths_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
base_path = 'opencompass/needlebench'
needle_file_name = 'needles.jsonl'

# Define configurations for both English and Chinese datasets
language_configs = [
    {
        'file_list': ['PaulGrahamEssays.jsonl'],
        'dataset_var': 'needlebench_en_datasets',
        'language': 'English',
        'length_buffer': 3000,
        'suffix': 'en'
    },
    {
        'file_list': ['zh_finance.jsonl'],
        'dataset_var': 'needlebench_zh_datasets',
        'language': 'Chinese',
        'length_buffer': 200,
        'suffix': 'zh'
    }
]

# Initialize empty dataset lists
needlebench_en_datasets = []
needlebench_zh_datasets = []

# Single loop to handle both languages
for config in language_configs:
    for original_context_length in context_lengths:
        for depth_percent in depths_list:
            dataset_dict = {
                'abbr': f'Length{original_context_length}'
                f'Depth{int(depth_percent)}_origin_{config["suffix"]}_1000k',
                'type': NeedleBenchOriginDataset,
                'path': base_path,
                'length': original_context_length,
                'depth': int(depth_percent),
                'tokenizer_model': 'gpt-4',
                'file_list': config['file_list'],
                'num_repeats_per_file': 10,
                'length_buffer': config['length_buffer'],
                'language': config['language'],
                'needle_file_name': needle_file_name,
                'reader_cfg': needlebench_reader_cfg,
                'infer_cfg': needlebench_infer_cfg,
                'eval_cfg': needlebench_eval_cfg,
            }
            globals()[config['dataset_var']].append(dataset_dict)
