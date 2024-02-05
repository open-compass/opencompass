from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HumanevalMultiDataset, HumanevalMultiEvaluator

humaneval_multi_reader_cfg = dict(input_columns=['prompt'], output_column='tests')

humaneval_multi_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template='{prompt}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024),
)

humaneval_multi_eval_cfg = {
    lang: dict(
        evaluator=dict(
            type=HumanevalMultiEvaluator,
            language=lang,
            ip_address='localhost',  # replace to your code_eval_server ip_address, port
            port=5000,
        ),  # refer to https://opencompass.readthedocs.io/en/latest/advanced_guides/code_eval_service.html to launch a server
        pred_role='BOT',
    ) for lang in ['cpp', 'cs', 'd', 'go', 'java', 'jl', 'js', 'lua', 'php', 'pl', 'py', 'r', 'rb', 'rkt', 'rs', 'scala', 'sh', 'swift', 'ts']
}

'''there are four versions of humaneval-{LANG}-{version}.jsonl:
['keep', 'transform', 'reworded', 'remove']
SRCDATA-LANG-keep is the same as SRCDATA-LANG, but the text of the prompt is totally unchanged. If the original prompt had Python doctests, they remain as Python instead of being translated to LANG. If the original prompt had Python-specific terminology, e.g., 'list', it remains 'list', instead of being translated, e.g., to 'vector' for C++.
SRCDATA-LANG-transform transforms the doctests to LANG but leaves the natural language text of the prompt unchanged.
SRCDATA-LANG-reworded transforms both the doctests and the natural language text of the prompt to LANG.
SRCDATA-LANG-remove removes the doctests from the prompt.
'''

humaneval_multi_datasets = [
    dict(
        type=HumanevalMultiDataset,
        abbr=f'humaneval_multiple-{lang}',
        language=lang,
        version='reworded',  # choose from ['keep', 'transform', 'reworded', 'remove']
        num_repeats=1,
        path='./data/multi-data/humaneval_multipl-e/',
        reader_cfg=humaneval_multi_reader_cfg,
        infer_cfg=humaneval_multi_infer_cfg,
        eval_cfg=humaneval_multi_eval_cfg[lang],
    ) for lang in ['cpp', 'cs', 'd', 'go', 'java', 'jl', 'js', 'lua', 'php', 'pl', 'py', 'r', 'rb', 'rkt', 'rs', 'scala', 'sh', 'swift', 'ts']
]
