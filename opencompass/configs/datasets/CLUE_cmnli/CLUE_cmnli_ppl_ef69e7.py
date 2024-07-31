from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CMNLIDataset

cmnli_reader_cfg = dict(
    input_columns=['sentence1', 'sentence2'],
    output_column='label',
    test_split='train')

cmnli_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'contradiction':
            dict(round=[
                dict(
                    role='HUMAN',
                    prompt='阅读文章：{sentence1}\n根据上文，回答如下问题：{sentence2}？'),
                dict(role='BOT', prompt='错')
            ]),
            'entailment':
            dict(round=[
                dict(
                    role='HUMAN',
                    prompt='阅读文章：{sentence1}\n根据上文，回答如下问题：{sentence2}？'),
                dict(role='BOT', prompt='对')
            ]),
            'neutral':
            dict(round=[
                dict(
                    role='HUMAN', prompt='如果{sentence1}为真，那么{sentence2}也为真吗？'),
                dict(role='BOT', prompt='可能')
            ]),
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

cmnli_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

cmnli_datasets = [
    dict(
        abbr='cmnli',
        type=CMNLIDataset,
        path='opencompass/cmnli-dev',
        reader_cfg=cmnli_reader_cfg,
        infer_cfg=cmnli_infer_cfg,
        eval_cfg=cmnli_eval_cfg)
]
