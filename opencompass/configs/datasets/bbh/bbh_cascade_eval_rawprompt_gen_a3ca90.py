"""
Summary: A cascade evaluation config for BBH free-form tasks.
Setting:
    Shot: 0-shot
    Evaluator:
        - CascadeEvaluator
            - BBHEvaluator
            - GenericLLMEvaluator
"""

from opencompass.datasets import (BBHDataset, BBHEvaluator,
                                  generic_llmjudge_postprocess)
from opencompass.evaluator import CascadeEvaluator, GenericLLMEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

bbh_reader_cfg = dict(input_columns=['input'], output_column='target')

bbh_free_form_sets = [
    'multistep_arithmetic_two',
    'navigate',
    'dyck_languages',
    'word_sorting',
    'sports_understanding',
    'boolean_expressions',
    'object_counting',
    'formal_fallacies',
    'causal_judgement',
    'web_of_lies',
]

bbh_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[{
            'role':
            'user',
            'content':
            'Follow the given examples and answer the question.\n\n'
            'Question: {input}\n You must give your final answer by '
            "starting with 'So the answer is' "
        }],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

GRADER_TEMPLATE = """
Please judge whether the predicted answer is consistent with the gold target
for the original question. The gold target is correct. Do not solve the
question yourself. Treat differences in capitalization, punctuation, or list
separators as equivalent when they express the same answer. A prediction with
additional explanation is correct only if its final answer agrees with the
gold target.

Reply with exactly one letter:
A: CORRECT
B: INCORRECT

<Original Question Begin>
{input}
<Original Question End>

<Gold Target Begin>
{target}
<Gold Target End>

<Predicted Answer Begin>
{prediction}
<Predicted Answer End>
""".strip()

cascade_evaluator = dict(
    type=CascadeEvaluator,
    rule_evaluator=dict(type=BBHEvaluator),
    llm_evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a precise answer evaluator.'
                },
                {
                    'role': 'user',
                    'content': GRADER_TEMPLATE
                },
            ],
        ),
        dataset_cfg=dict(
            type=BBHDataset,
            path='opencompass/bbh',
            reader_cfg=bbh_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),
    ),
    parallel=False,
)

bbh_datasets = []
for _name in bbh_free_form_sets:
    bbh_datasets.append(
        dict(
            type=BBHDataset,
            path='opencompass/bbh',
            name=_name,
            abbr='bbh-' + _name,
            reader_cfg=bbh_reader_cfg,
            infer_cfg=bbh_infer_cfg.copy(),
            eval_cfg=dict(evaluator=cascade_evaluator.copy()),
        ))
