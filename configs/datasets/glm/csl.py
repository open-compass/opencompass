from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GLMChoiceInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CslDataset

csl_reader_cfg = dict(
    input_columns=["abst", "keywords"], output_column='label')

csl_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template={
            0: "</E>摘要：</A>",
            1: "</E>摘要：</A>关键词：</K>"
        },
        column_token_map={
            "abst": '</A>',
            'keywords': '</K>'
        },
        ice_token='</E>'),
    prompt_template=dict(
        type=PromptTemplate,
        template=
        '</E>Abstract: </A>\nKeyword: </K>\n Does all keywords come from the given abstract? (Yes or No)',
        column_token_map={
            "abst": '</A>',
            'keywords': '</K>'
        },
        ice_token='</E>'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GLMChoiceInferencer, choices=['No', 'Yes']))

csl_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

csl_datasets = [
    dict(
        type=CslDataset,
        path='json',
        abbr='csl',
        data_files='./data/FewCLUE/csl/test_public.json',
        split='train',
        reader_cfg=csl_reader_cfg,
        infer_cfg=csl_infer_cfg,
        eval_cfg=csl_eval_cfg)
]
