# base config for LLM4Chem
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import PEER_postprocess, PEER_Evaluator, PEER_Dataset, PEER_postprocess_float_compare, \
    PEER_postprocess_default

TASKS = [
    'solubility',
    'stability',
    'human_ppi',
    'yeast_ppi',
]

reader_cfg = dict(input_columns=['input'], output_column='output')

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='{input}.'),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer,
        # max_out_len=2048,
    )
)

eval_cfg = dict(
    evaluator=dict(type=PEER_Evaluator),
    pred_postprocessor=dict(type=PEER_postprocess),
    dataset_postprocessor=dict(type=PEER_postprocess),
)

# use default postprocess to remain the original output for LLM judgement.
# PEER_postprocess will be used in the evaluation stage to compare the output with the ground truth as a fast comparison.
eval_llm_cfg = dict(
    evaluator=dict(type=PEER_Evaluator,
                   openai_key='EMPTY', gpt_model='gpt-4.1-mini'),
    pred_postprocessor=dict(type=PEER_postprocess_default),
    dataset_postprocessor=dict(type=PEER_postprocess_default),
)

eval_stability_cfg = dict(
    evaluator=dict(type=PEER_Evaluator, task='stability'),
    pred_postprocessor=dict(type=PEER_postprocess_float_compare, compare_number=1),
    dataset_postprocessor=dict(type=PEER_postprocess_float_compare, compare_number=1),
)

PEER_datasets = []
mini_PEER_datasets = []

for task in TASKS:
    if task != 'stability':
        PEER_datasets.append(
            dict(
                abbr=f'SciReasoner-PEER_{task}',
                type=PEER_Dataset,
                path='opencompass/SciReasoner-PEER',
                task=task,
                reader_cfg=reader_cfg,
                infer_cfg=infer_cfg,
                eval_cfg=eval_llm_cfg),
        )
        mini_PEER_datasets.append(
            dict(
                abbr=f'SciReasoner-PEER_{task}-mini',
                type=PEER_Dataset,
                path='opencompass/SciReasoner-PEER',
                task=task,
                mini_set=True,
                reader_cfg=reader_cfg,
                infer_cfg=infer_cfg,
                eval_cfg=eval_llm_cfg),
        )
    else:
        PEER_datasets.append(
            dict(
                abbr=f'SciReasoner-PEER_{task}',
                type=PEER_Dataset,
                path='opencompass/SciReasoner-PEER',
                task=task,
                reader_cfg=reader_cfg,
                infer_cfg=infer_cfg,
                eval_cfg=eval_stability_cfg),
        )
        mini_PEER_datasets.append(
            dict(
                abbr=f'SciReasoner-PEER_{task}-mini',
                type=PEER_Dataset,
                path='opencompass/SciReasoner-PEER',
                task=task,
                mini_set=True,
                reader_cfg=reader_cfg,
                infer_cfg=infer_cfg,
                eval_cfg=eval_stability_cfg),
        )
