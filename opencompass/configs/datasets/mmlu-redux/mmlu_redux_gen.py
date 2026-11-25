"""MMLU-Redux: a manually re-annotated subset of MMLU.

Source: https://github.com/aryopg/mmlu-redux, HF dataset fxmarty/mmlu-redux-2.0-ok.
Generative protocol mirrors the lm-evaluation-harness mmlu-redux generative tasks:
the model must answer with a single letter (A-D), which is extracted and scored
with exact match. Requested in #2132.
"""

from opencompass.datasets import MMLUReduxDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.utils.text_postprocessors import match_answer_pattern

SUBJECTS = [
    'abstract_algebra','anatomy','astronomy','business_ethics','clinical_knowledge','college_biology','college_chemistry','college_computer_science','college_mathematics','college_medicine','college_physics','computer_security','conceptual_physics','econometrics','electrical_engineering','elementary_mathematics','formal_logic','global_facts','high_school_biology','high_school_chemistry','high_school_computer_science','high_school_european_history','high_school_geography','high_school_government_and_politics','high_school_macroeconomics','high_school_mathematics','high_school_microeconomics','high_school_physics','high_school_psychology','high_school_statistics','high_school_us_history','high_school_world_history','human_aging','human_sexuality','international_law','jurisprudence','logical_fallacies','machine_learning','management','marketing','medical_genetics','miscellaneous','moral_disputes','moral_scenarios','nutrition','philosophy','prehistory','professional_accounting','professional_law','professional_medicine','professional_psychology','public_relations','security_studies','sociology','us_foreign_policy','virology','world_religions'
]

QUERY_TEMPLATE = (
    '{question}\n'
    'A. {A}\n'
    'B. {B}\n'
    'C. {C}\n'
    'D. {D}\n'
    'Please respond with the correct letter (A, B, C or D) without any '
    'additional comments, only the correct letter:'
)

mmlu_redux_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='target',
)

mmlu_redux_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(role='HUMAN', prompt=QUERY_TEMPLATE)],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

mmlu_redux_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=match_answer_pattern, answer_pattern=r'(?i)^\s*\(?([A-D])\)?'),
)

mmlu_redux_datasets = []
for _subject in SUBJECTS:
    mmlu_redux_datasets.append(
        dict(
            abbr=f'mmlu_redux_{_subject}',
            type=MMLUReduxDataset,
            path='fxmarty/mmlu-redux-2.0-ok',
            name=_subject,
            reader_cfg=mmlu_redux_reader_cfg,
            infer_cfg=mmlu_redux_infer_cfg,
            eval_cfg=mmlu_redux_eval_cfg,
        )
    )
