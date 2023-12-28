from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.cdme.cdme import CDMEDataset,CDMEEvaluator,cdme_postprocess,cdme_dataset_postprocess
import math


def logistic(x, L=100, x0=50, k=0.1):
    return round(L / (1 + math.exp(-k * (x - x0))), 3)


def generate_linear_space(start, end, num):
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

cdme_reader_cfg = dict(input_columns=['prompt'], output_column='answer')

cdme_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='''{prompt}'''),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

cdme_eval_cfg = dict(
    evaluator=dict(type=CDMEEvaluator),
    pred_postprocessor=dict(type=cdme_postprocess),
    dataset_postprocessor=dict(type=cdme_dataset_postprocess),
    pred_role='BOT')

context_lengths = list(range(1000, 9000, 1000))
document_depth_percent_intervals = 35
document_depth_percent_interval_type = "linear"

base_path = './data/CDME/processed'
cdme_datasets = []


for original_context_length in context_lengths:
    for depth_percent in generate_depth_percents(
            document_depth_percent_intervals,
            document_depth_percent_interval_type):
        dataset_dict = dict(
            abbr=f'CDME_Length{original_context_length}'
            'Depth{int(depth_percent)}',
            type=CDMEDataset,
            path=base_path,
            length=original_context_length,
            depth=int(depth_percent),
            reader_cfg=cdme_reader_cfg,
            infer_cfg=cdme_infer_cfg,
            eval_cfg=cdme_eval_cfg
        )
        cdme_datasets.append(dataset_dict)
