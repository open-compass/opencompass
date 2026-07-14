from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import CsimpleqaDataset, csimpleqa_postprocess

subjective_reader_cfg = dict(input_columns=['primary_category', 'question','gold_ans', 'messages', 'system_prompt','prompt_template'], output_column='judge')

subjective_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': '{question}'},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

subjective_eval_cfg = dict(
    evaluator=dict(
        type=LMEvaluator,
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                {'role': 'system', 'content': '{system_prompt}'},
                {'role': 'user', 'content': '{prompt_template}'}
            ],
        ),
        dict_postprocessor=dict(type=csimpleqa_postprocess),
    ),
    pred_role='BOT',
)

csimpleqa_datasets = [
    dict(
        abbr='chinese_simpleqa',
        type=CsimpleqaDataset,
        name='chinese_simpleqa',
        path='opencompass/chinese_simpleqa',
        reader_cfg=subjective_reader_cfg,
        infer_cfg=subjective_infer_cfg,
        eval_cfg=subjective_eval_cfg,
        mode='singlescore',
    )
]
