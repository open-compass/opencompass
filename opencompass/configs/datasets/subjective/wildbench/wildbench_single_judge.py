from opencompass.datasets import WildBenchDataset
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

wildbench_reader_cfg = dict(
    input_columns=['dialogue', 'prompt'],
    output_column='judge',
)

data_path ='/mnt/hwfile/opendatalab/yanghaote/datasets/WildBench/wildbench.jsonl'

wildbench_datasets = []
wildbench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""{dialogue}"""
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=ChatInferencer,
        max_seq_len=32768,
        max_out_len=8192,
        infer_mode='last',
    ),
)

wildbench_eval_cfg = dict(
    evaluator=dict(
        type=LMEvaluator,
        prompt_template=dict(
            type=PromptTemplate, 
            template="""{prompt}"""
        ),
    ),
    pred_role='BOT',
)

wildbench_datasets.append(
    dict(
        abbr='wildbench',
        type=WildBenchDataset,
        path=data_path,
        eval_mode='single',
        reader_cfg=wildbench_reader_cfg,
        infer_cfg=wildbench_infer_cfg,
        eval_cfg=wildbench_eval_cfg,
    )
)