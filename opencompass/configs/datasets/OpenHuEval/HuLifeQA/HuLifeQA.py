from opencompass.datasets import WildBenchDataset
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

with read_base():
    from .HuLifeQA_setting import DATA_PATH, TASK_GROUP_NEW

hu_life_qa_reader_cfg = dict(
    input_columns=['dialogue', 'prompt'],
    output_column='judge',
)

hu_life_qa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""{dialogue}"""
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=ChatInferencer,
        max_seq_len=8192,
        max_out_len=8192,
        infer_mode='last',
    ),
)

hu_life_qa_eval_cfg = dict(
    evaluator=dict(
        type=LMEvaluator,
        prompt_template=dict(
            type=PromptTemplate, 
            template="""{prompt}"""
        ),
    ),
    pred_role='BOT',
)

hu_life_qa_datasets = []
hu_life_qa_datasets.append(
    dict(
        abbr='open_hu_eval_hu_life_qa',
        type=WildBenchDataset,
        path=DATA_PATH,
        reader_cfg=hu_life_qa_reader_cfg,
        infer_cfg=hu_life_qa_infer_cfg,
        eval_cfg=hu_life_qa_eval_cfg,
    )
)

