from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset

ocnli_reader_cfg = dict(
    input_columns=['sentence1', 'sentence2'], output_column='label')

# TODO: two prompt templates for ocnli
ocnli_infer_cfg = dict(
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

ocnli_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )

ocnli_datasets = [
    dict(
        type=HFDataset,
        abbr='ocnli',
        path='json',
        split='train',
        data_files='./data/CLUE/OCNLI/dev.json',
        reader_cfg=ocnli_reader_cfg,
        infer_cfg=ocnli_infer_cfg,
        eval_cfg=ocnli_eval_cfg)
]
