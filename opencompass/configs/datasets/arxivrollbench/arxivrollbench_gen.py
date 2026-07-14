from opencompass.datasets import HFDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

_DOMAINS = [
    ('cs', 'cs'),
    ('q_fin', 'q-fin'),
    ('math', 'math'),
    ('physics', 'physics'),
    ('stat', 'stat'),
    ('q_bio', 'q-bio'),
    ('econ', 'econ'),
    ('eess', 'eess'),
]
_RELEASES = ['2024b', '2025a', '2026a']
_TASK_TYPES = ['s', 'c', 'p']

_S_PROMPT = """## Instruction:
 Given a **shuffled text** composed of sentences A, B, and C, your task is to select the correct order from four available selections. Avoid providing any additional information (such as explanations of your choice) or restating the sentences in your answer. Simply provide your selection: Selection 1, Selection 2, Selection 3, or Selection 4.
## Shuffled text:
{shuffled_text}
## Choice:
**Selection 1** {A}
**Selection 2** {B}
**Selection 3** {C}
**Selection 4** {D}
Answer:"""

_C_PROMPT = """## Instruction:
 Given a **masked paragraph** with three masked sentences marked as '<|MaskedSentence|>' and candidate sentences labeled A, B, and C, your task is to fill in the correct sentences to the masked positions by selecting the appropriate answers from four provided selections. Avoid providing any additional information (such as explanations of your choice) or restating the sentences in your answer. Simply provide your selection: Selection 1, Selection 2, Selection 3, or Selection 4.
## Masked paragraph:
{text_with_holes}
## {text_candidates}
 ## Choice:
**Selection 1** {A}
**Selection 2** {B}
**Selection 3** {C}
**Selection 4** {D}
Answer:"""

_P_PROMPT = """## Instruction:
 Given a context, and four choices marked as A, B, C, and D, your task is to select the correct text which is the next sequence of the provided context. Avoid providing any additional information (such as explanations of your choice) or restating the choice in your answer. Simply provide one of the four letters: A, B, C, or D.
## Context:
{context}
## Choice:
**A** {A}
**B** {B}
**C** {C}
**D** {D}
Answer:"""


def _dataset_path(release, hf_domain, task_type, compact=True):
    suffix = '-50' if compact else ''
    if release == '2024b':
        return f'liangzid/robench2024b_all_set{hf_domain}SCP-{task_type}{suffix}'
    return (f'liangzid/robench{release}_test_all_category_set'
            f'{hf_domain}SCP-{task_type}{suffix}')


def _reader_cfg(task_type):
    if task_type == 's':
        return dict(
            input_columns=['shuffled_text', 'A', 'B', 'C', 'D'],
            output_column='label',
            train_split='train',
            test_split='train',
        )
    if task_type == 'c':
        return dict(
            input_columns=['text_with_holes', 'text_candidates', 'A', 'B',
                           'C', 'D'],
            output_column='label',
            train_split='train',
            test_split='train',
        )
    return dict(
        input_columns=['context', 'A', 'B', 'C', 'D'],
        output_column='label',
        train_split='train',
        test_split='train',
    )


def _infer_cfg(task_type):
    prompt = dict(s=_S_PROMPT, c=_C_PROMPT, p=_P_PROMPT)[task_type]
    return dict(
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                dict(role='user', content=prompt),
            ],
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )


def _eval_cfg(task_type):
    postprocessor = ('arxivrollbench_selection_postprocess' if task_type
                     in ['s', 'c'] else 'arxivrollbench_choice_postprocess')
    return dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(type=postprocessor),
    )


def _build_datasets(compact=True):
    datasets = []
    for release in _RELEASES:
        for domain, hf_domain in _DOMAINS:
            for task_type in _TASK_TYPES:
                suffix = '' if compact else '-full'
                datasets.append(
                    dict(
                        abbr=(f'arxivrollbench-{release}-{domain}-'
                              f'{task_type}{suffix}'),
                        type=HFDataset,
                        path=_dataset_path(release, hf_domain, task_type,
                                           compact),
                        reader_cfg=_reader_cfg(task_type),
                        infer_cfg=_infer_cfg(task_type),
                        eval_cfg=_eval_cfg(task_type),
                    ))
    return datasets


# Default ArxivRollBench configuration: compact 50-sample splits.
arxivrollbench_datasets = _build_datasets(compact=True)

# Full public splits are also provided for complete benchmark runs.
arxivrollbench_full_datasets = _build_datasets(compact=False)
