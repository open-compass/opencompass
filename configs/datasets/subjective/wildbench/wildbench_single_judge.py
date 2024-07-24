from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import ChatInferencer, GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import WildBenchDataset


subjective_reader_cfg = dict(
    input_columns=['dialogue', 'prompt'],
    output_column='judge',
    )

data_path ='./data/WildBench/wildbench.jsonl'

wildbench_single_datasets = []

# the question is a list, how to process it
subjective_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template="""{dialogue}"""
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=ChatInferencer, max_seq_len=4096, max_out_len=512, infer_mode='last'),
    )

subjective_eval_cfg = dict(
    evaluator=dict(
        type=LMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template="""{prompt}"""
        ),
    ),
    pred_role='BOT',
)

wildbench_single_datasets.append(
    dict(
        abbr='wildbench',
        type=WildBenchDataset,
        path=data_path,
        eval_mode='single',
        reader_cfg=subjective_reader_cfg,
        infer_cfg=subjective_infer_cfg,
        eval_cfg=subjective_eval_cfg
    ))
