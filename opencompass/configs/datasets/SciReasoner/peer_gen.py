# base config for LLM4Chem
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import PEER_postprocess, PEER_Evaluator, PEER_Dataset, PEER_postprocess_float_compare, \
    PEER_postprocess_default, PEERRuleEvaluator, peer_llm_judge_postprocess
from opencompass.evaluator import (
    CascadeEvaluator,
    GenericLLMEvaluator,
)

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
    evaluator=dict(type=PEER_Evaluator),
    pred_postprocessor=dict(type=PEER_postprocess_default),
    dataset_postprocessor=dict(type=PEER_postprocess_default),
)

JUDGE_TEMPLATE = """
Please determine whether this answer is correct. Definition: 'Correct': The core conclusion of the model's answer (if any) is completely consistent with the reference answer (literal identity is not required). 'Incorrect': The core conclusion of the model's answer is consistent with the reference answer, or the core conclusion is not clearly expressed.
Reference answer: {reference}
Model answer: {prediction}
If correct, answer 'True'; if incorrect, answer 'False'. Please only answer 'True' or 'False'.
""".strip()



eval_stability_cfg = dict(
    evaluator=dict(type=PEER_Evaluator, task='stability'),
    pred_postprocessor=dict(type=PEER_postprocess_float_compare, compare_number=1),
    dataset_postprocessor=dict(type=PEER_postprocess_float_compare, compare_number=1),
)


PEER_datasets = []
mini_PEER_datasets = []

for task in TASKS:

    peer_llm_evaluator_cfg = dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs."
                    )
                ],
                round=[
                    dict(role='HUMAN', prompt=JUDGE_TEMPLATE),
                ],
            ),
        ),
        dataset_cfg=dict(
            type=PEER_Dataset,
            path='opencompass/SciReasoner-PEER',
            task=task,
            reader_cfg=reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=peer_llm_judge_postprocess),
    )

    peer_rule_evaluator_cfg = dict(
        type=PEERRuleEvaluator
    )

    cascade_evaluator = dict(
        type=CascadeEvaluator,
        rule_evaluator=peer_rule_evaluator_cfg,
        llm_evaluator=peer_llm_evaluator_cfg,
        parallel=False,
    )

    cascade_eval_llm_cfg = dict(
        evaluator=cascade_evaluator,
        pred_postprocessor=dict(type=PEER_postprocess_default),
        dataset_postprocessor=dict(type=PEER_postprocess_default),
    )




    PEER_datasets.append(
        dict(
            abbr=f'SciReasoner-PEER_{task}',
            type=PEER_Dataset,
            path='opencompass/SciReasoner-PEER',
            task=task,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=cascade_eval_llm_cfg),
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
            eval_cfg=cascade_eval_llm_cfg),
    )

