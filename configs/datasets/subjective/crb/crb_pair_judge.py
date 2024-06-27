from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import ChatInferencer, GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import CRBDataset


subjective_reader_cfg = dict(
    input_columns=['question', 'prompt'],
    output_column='judge',
    )


data_path ='/mnt/petrelfs/zhangchuyu/reasoning/compasssak/crb/crbbenchv2.jsonl'

subjective_datasets = []
subjective_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt='{question}'
                    ),
                ]),
            ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_seq_len=4096, max_out_len=4096),
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

subjective_datasets.append(
    dict(
        abbr='crbbench',
        type=CRBDataset,
        path=data_path,
        mode='pair',
        reader_cfg=subjective_reader_cfg,
        infer_cfg=subjective_infer_cfg,
        eval_cfg=subjective_eval_cfg
    ))
