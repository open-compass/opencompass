from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator
from opencompass.datasets import TydiqaIdDataset

tydiqaId_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='answer',
    test_split='test')


tydiqaId_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Read the given Indonisian context and answer the following question. Your response should be in Indonisian and as simple as possible. Context: {context} Question: {question}?'),
                dict(role='BOT', prompt='Answer:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=50))

tydiqaId_eval_cfg = dict(
    evaluator=dict(type=EMEvaluator),
    pred_role="BOT")

tydiqaId_datasets = [
    dict(
        type=TydiqaIdDataset,
        abbr='tydiqa_id',
        path='./data/tydiqa_id/',
        reader_cfg=tydiqaId_reader_cfg,
        infer_cfg=tydiqaId_infer_cfg,
        eval_cfg=tydiqaId_eval_cfg)
]