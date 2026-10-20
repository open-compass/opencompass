# Select the 10 most popular programming languages from MultiPL-E to compose the test set.

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MultiplEDataset, MultiplEEvaluator


_TOP_TEN_LANGUAGE_ = ['cpp']

multiple_reader_cfg = dict(input_columns=['language', 'prompt'], output_column='tests')

multiple_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template='Based on the provided {language} code snippet, complete the subsequent content. The initial part of the completed code must match the provided code snippet exactly:\n{prompt}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

multiple_eval_cfg = {
    lang: dict(
        evaluator=dict(
            type=MultiplEEvaluator,
            language=lang,
            ip_address='https://opencompass-multiple-evaluator.hf.space',
        ),
        pred_role='BOT',
    ) for lang in _TOP_TEN_LANGUAGE_
}

multiple_datasets = [
    dict(
        type=MultiplEDataset,
        abbr=f'humaneval-multiple-{lang}',
        language=lang,
        path='opencompass/multipl_e',
        tag='humaneval',
        reader_cfg=multiple_reader_cfg,
        infer_cfg=multiple_infer_cfg,
        eval_cfg=multiple_eval_cfg[lang],
        n=5,
        k=3
    ) for lang in _TOP_TEN_LANGUAGE_
]

multiple_datasets += [
    dict(
        type=MultiplEDataset,
        abbr=f'mbpp-multiple-{lang}',
        language=lang,
        path='opencompass/multipl_e',
        tag='mbpp',
        reader_cfg=multiple_reader_cfg,
        infer_cfg=multiple_infer_cfg,
        eval_cfg=multiple_eval_cfg[lang],
        n=5,
        k=3
    ) for lang in _TOP_TEN_LANGUAGE_
]
