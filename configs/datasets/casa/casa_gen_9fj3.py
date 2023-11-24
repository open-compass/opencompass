from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CasaDataset, CasaEvaluator



casa_reader_cfg = dict(
    input_columns=['text'],
    output_column='label',
    test_split='test')

casa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt="Read each review carefully and assign one label from the set of ('positive', 'neutral', 'negative') for each of the provided categories: 'fuel', 'machine', 'others', 'part', 'price', 'service', ensuring that each label accurately reflects the content and sentiment of the review and seperate them with ', '. Review: {text}?"),
                dict(role='BOT', prompt='Label:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=50))

casa_eval_cfg = dict(
    evaluator=dict(type=CasaEvaluator),
    pred_role="BOT")


casa_datasets = [
    dict(
        type=CasaDataset,
        abbr='casa',
        path='./data/casa/',
        reader_cfg=casa_reader_cfg,
        infer_cfg=casa_infer_cfg,
        eval_cfg=casa_eval_cfg)
]


