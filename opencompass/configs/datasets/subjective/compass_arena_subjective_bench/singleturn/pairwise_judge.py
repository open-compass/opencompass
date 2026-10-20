from mmengine.config import read_base

from opencompass.datasets import (
    CompassArenaSubjectiveBench,
    compassarena_subjectiveeval_pairwise_postprocess,
)
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

subjective_reader_cfg = dict(
    input_columns=['question', 'pairwise_judge_prompt'],
    output_column='judge',
)

subjective_all_sets = [
    'singleturn',
]

qwen_2_5_72b = [
    dict(
        abbr='Qwen-2.5-72B-Instruct',
    )
]

compassarena_subjectivebench_singleturn_datasets = []


for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt='{question}'),
                ]
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=4096),
    )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='{pairwise_judge_prompt}'),
                    ]
                ),
            ),
            dict_postprocessor=dict(
                type=compassarena_subjectiveeval_pairwise_postprocess
            ),
        ),
        pred_role='BOT',
    )

    compassarena_subjectivebench_singleturn_datasets.append(
        dict(
            abbr=f'{_name}',
            type=CompassArenaSubjectiveBench,
            path='./data/subjective/CompassArenaSubjectiveBench',
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg,
            mode='m2n',
            infer_order='double',
            base_models=qwen_2_5_72b,
            given_pred=[
                {
                    'abbr': 'Qwen-2.5-72B-Instruct',
                    'path': './data/subjective/CompassArenaSubjectiveBench/Qwen-2.5-72B-Instruct',
                }
            ],
        )
    )
