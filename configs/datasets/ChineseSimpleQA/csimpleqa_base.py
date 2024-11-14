from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, ChatInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import CsimpleqaDataset, CsimpleqaEvaluator, qa_base_preprocess

csimpleqa_reader_cfg = dict(input_columns=['question','gold_ans', 'messages', 'system_prompt','prompt_template'], output_column='gold_ans')

csimpleqa_base_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='问题：姚明是什么运动员？\n回答：篮球运动员\n问题：北京鸟巢是哪一年完工的？\n回答：2008年\n问题：{question}\n回答：',
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=50)
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
    pred_postprocessor = dict(type=qa_base_preprocess),
    pred_role='BOT',
)

csimpleqa_base_datasets = [
    dict(
        abbr='csimpleqa',
        type=CsimpleqaDataset,
        path='./data/csimpleqa',
        reader_cfg=csimpleqa_reader_cfg,
        infer_cfg=csimpleqa_base_infer_cfg,
        eval_cfg=csimpleqa_eval_cfg
    )
]
