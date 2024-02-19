from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CslDataset

csl_reader_cfg = dict(
    input_columns=['abst', 'keywords'], output_column='label')

csl_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0: '摘要：{abst}',
            1: '摘要：{abst}\n关键词：{keywords}'
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

csl_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

csl_datasets = [
    dict(
        type=CslDataset,
        path='json',
        abbr='csl_dev',
        data_files='./data/FewCLUE/csl/dev_few_all.json',
        split='train',
        reader_cfg=csl_reader_cfg,
        infer_cfg=csl_infer_cfg,
        eval_cfg=csl_eval_cfg),
    dict(
        type=CslDataset,
        path='json',
        abbr='csl_test',
        data_files='./data/FewCLUE/csl/test_public.json',
        split='train',
        reader_cfg=csl_reader_cfg,
        infer_cfg=csl_infer_cfg,
        eval_cfg=csl_eval_cfg)
]
