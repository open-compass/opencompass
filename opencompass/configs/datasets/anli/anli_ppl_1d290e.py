from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import AnliDataset

anli_datasets = []
for _split in ['R1', 'R2', 'R3']:
    anli_reader_cfg = dict(
        input_columns=['context', 'hypothesis'],
        output_column='label',
    )

    anli_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template={
                'A':
                dict(round=[
                    dict(role='HUMAN', prompt='{context}\n{hypothesis}\What is the relation between the two sentences?'),
                    dict(role='BOT', prompt='Contradiction'),
                ]),
                'B':
                dict(round=[
                    dict(role='HUMAN', prompt='{context}\n{hypothesis}\What is the relation between the two sentences?'),
                    dict(role='BOT', prompt='Entailment'),
                ]),
                'C':
                dict(round=[
                    dict(role='HUMAN', prompt='{context}\n{hypothesis}\What is the relation between the two sentences?'),
                    dict(role='BOT', prompt='Neutral'),
                ]),
            },
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=PPLInferencer),
    )

    anli_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )

    anli_datasets.append(
        dict(
            type=AnliDataset,
            abbr=f'anli-{_split}',
            path=f'data/anli/anli_v1.0/{_split}/dev.jsonl',
            reader_cfg=anli_reader_cfg,
            infer_cfg=anli_infer_cfg,
            eval_cfg=anli_eval_cfg,
        )
    )
