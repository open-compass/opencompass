from opencompass.datasets import HFDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
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

_SC_PROMPT = """You are evaluating an ArxivRollBench sequencing or cloze task.
Choose the correct option and answer only with Selection 1, Selection 2, Selection 3, or Selection 4.

{shuffled_text}

Selection 1: {A}
Selection 2: {B}
Selection 3: {C}
Selection 4: {D}

Answer:"""

_P_PROMPT = """You are evaluating an ArxivRollBench prediction task.
Choose the correct option and answer only with A, B, C, or D.

Context:
{context}

A. {A}
B. {B}
C. {C}
D. {D}

Answer:"""


def _dataset_path(release, hf_domain, task_type, compact=True):
    suffix = '-50' if compact else ''
    if release == '2024b':
        return f'liangzid/robench2024b_all_set{hf_domain}SCP-{task_type}{suffix}'
    return (f'liangzid/robench{release}_test_all_category_set'
            f'{hf_domain}SCP-{task_type}{suffix}')


def _reader_cfg(task_type):
    if task_type in ['s', 'c']:
        return dict(
            input_columns=['shuffled_text', 'A', 'B', 'C', 'D'],
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
    prompt = _SC_PROMPT if task_type in ['s', 'c'] else _P_PROMPT
    return dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[dict(role='HUMAN', prompt=prompt)]),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=50),
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
