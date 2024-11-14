from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, ChatInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import CsimpleqaDataset, CsimpleqaEvaluator

csimpleqa_reader_cfg = dict(input_columns=['question','gold_ans', 'messages', 'system_prompt','prompt_template'], output_column='prediction')

csimpleqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""{messages}""",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ChatInferencer, max_out_len=100, infer_mode="last")
)

csimpleqa_eval_cfg = dict(
    evaluator=dict(
        type=CsimpleqaEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
            begin=[
                dict(
                    role='SYSTEM',
                    prompt='{system_prompt}')
            ],
                round=[
                dict(
                    role='HUMAN',
                    prompt = '{prompt_template}'
                ),
            ]),
        ),
    ),
    pred_role='BOT',
)

csimpleqa_datasets = [
    dict(
        abbr='csimpleqa',
        type=CsimpleqaDataset,
        path='./data/csimpleqa',
        reader_cfg=csimpleqa_reader_cfg,
        infer_cfg=csimpleqa_infer_cfg,
        eval_cfg=csimpleqa_eval_cfg
    )
]
