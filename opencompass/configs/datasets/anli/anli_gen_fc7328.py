from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import AnliDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess

anli_datasets = []
for _split in ['R1', 'R2', 'R3']:
    anli_reader_cfg = dict(
        input_columns=['context', 'hypothesis'],
        output_column='label',
    )

    anli_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt='{context}\n{hypothesis}\nQuestion: What is the relation between the two sentences?\nA. Contradiction\nB. Entailment\nC. Neutral\nAnswer: '),
                    dict(role='BOT', prompt='{label}'),
                ]
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    anli_eval_cfg = dict(evaluator=dict(type=AccEvaluator),
                         pred_role='BOT',
                         pred_postprocessor=dict(type=first_capital_postprocess))

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
