from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets.subjectivity_cmp import SubjectivityCmpDataset

subjectivity_reader_cfg = dict(
    input_columns=['question', 'index', 'reference_answer', 'evaluating_guidance', 'capability', 'prompt'],
    output_column=None,
    train_split='test')

subjectivity_all_sets = [
    "sub_test",
]

subjectivity_datasets = []

for _name in subjectivity_all_sets:
    subjectivity_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt="{question}"
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        )

    subjectivity_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            cmp_order='both',
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role="SYSTEM",
                            fallback_role="HUMAN",
                            prompt="{prompt}"
                        ),
                    ],
                    round=[dict(role="HUMAN",
                                prompt="回答 1: <回答 1 开始> {prediction} <回答 1 结束>\n回答 2: <回答 2 开始> {prediction2} <回答 2 结束>\n")]))),
        pred_role="BOT",
    )

    subjectivity_datasets.append(
        dict(
            abbr=f"{_name}",
            type=SubjectivityCmpDataset,
            path="./data/subjectivity/",
            name=_name,
            reader_cfg=subjectivity_reader_cfg,
            infer_cfg=subjectivity_infer_cfg,
            eval_cfg=subjectivity_eval_cfg
        ))
