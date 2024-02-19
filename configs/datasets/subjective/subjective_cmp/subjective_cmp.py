from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets.subjective_cmp import SubjectiveCmpDataset

subjective_reader_cfg = dict(
    input_columns=['question', 'index', 'reference_answer', 'evaluating_guidance', 'capability', 'prompt'],
    output_column='judge',
    train_split='test')

subjective_all_sets = [
    'creation_v0.1',
]

subjective_datasets = []

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
            inferencer=dict(type=GenInferencer, max_out_len=2048),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            cmp_order='both',
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role='SYSTEM',
                            fallback_role='HUMAN',
                            prompt='{prompt}'
                        ),
                    ],
                    round=[dict(role='HUMAN',
                                prompt='回答 1: <回答 1 开始> {prediction} <回答 1 结束>\n回答 2: <回答 2 开始> {prediction2} <回答 2 结束>\n')]))),
        pred_role='BOT',
    )

    subjective_datasets.append(
        dict(
            abbr=f'{_name}',
            type=SubjectiveCmpDataset,
            path='./data/subjective/',
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg
        ))
