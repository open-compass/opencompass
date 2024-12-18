from mmengine.config import read_base

from opencompass.datasets import (  # compassarena_subjectiveeval_pairwise_postprocess,
    CompassArenaSubjectiveBench,
    compassarena_subjectiveeval_bradleyterry_postprocess,
)
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

subjective_reader_cfg = dict(
    input_columns=['dialogue', 'pairwise_judge_prompt'],
    output_column='judge',
)

subjective_all_sets = [
    'multiturn',
]

qwen_2_5_72b = [
    dict(
        abbr='Qwen-2.5-72B-Instruct',
    )
]

compassarena_subjectivebench_bradleyterry_multiturn_datasets = []


for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt='{dialogue}'),
                ]
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(
            type=ChatInferencer, max_seq_len=8192, max_out_len=2048, infer_mode='every'
        ),
    )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            pack_all_predictions=True,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='{pairwise_judge_prompt}'),
                    ]
                ),
            ),
            dict_postprocessor=dict(
                type=compassarena_subjectiveeval_bradleyterry_postprocess
            ),
            keep_predictions=True,  # Must be turned on to save predictions from model pairs to calculate style features in postprocessor
        ),
        pred_role='BOT',
    )

    compassarena_subjectivebench_bradleyterry_multiturn_datasets.append(
        dict(
            abbr=f'{_name}',
            type=CompassArenaSubjectiveBench,
            path='./data/subjective/CompassArenaSubjectiveBench',
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg,
            mode='m2n',
            infer_order='random',
            base_models=qwen_2_5_72b,
            given_pred=[
                {
                    'abbr': 'Qwen-2.5-72B-Instruct',
                    'path': './data/subjective/CompassArenaSubjectiveBench/Qwen-2.5-72B-Instruct',
                }
            ],
        )
    )
