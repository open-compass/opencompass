from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import CompassArenaSubjectiveBench, compassarena_subjectiveeval_pointwise_postprocess
from mmengine.config import read_base

subjective_reader_cfg = dict(
    input_columns=['question', 'pointwise_judge_prompt'],
    output_column='judge',
    )

subjective_all_sets = [
    'singleturn',
]


compassarena_subjectivebench_singleturn_datasets = []


for _name in subjective_all_sets:
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
            inferencer=dict(type=GenInferencer, max_out_len=4096),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                    dict(
                        role='HUMAN',
                        prompt = '{pointwise_judge_prompt}'
                    ),
                ]),
            ),
            dict_postprocessor=dict(type=compassarena_subjectiveeval_pointwise_postprocess),
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
            mode='singlescore',
        ))
