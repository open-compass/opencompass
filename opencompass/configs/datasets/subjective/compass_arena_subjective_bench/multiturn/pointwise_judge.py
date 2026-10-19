from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import CompassArenaSubjectiveBench, compassarena_subjectiveeval_pointwise_postprocess
from mmengine.config import read_base

subjective_reader_cfg = dict(
    input_columns=['dialogue', 'pointwise_judge_prompt'],
    output_column='judge',
    )

subjective_all_sets = [
    'multiturn',
]


compassarena_subjectivebench_multiturn_datasets = []


for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt='{dialogue}'
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=ChatInferencer, max_seq_len=8192, max_out_len=2048, infer_mode='every'),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            pack_all_predictions=True,
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

    compassarena_subjectivebench_multiturn_datasets.append(
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
