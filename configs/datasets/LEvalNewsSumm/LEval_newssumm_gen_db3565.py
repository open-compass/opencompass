from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator, RougeEvaluator, SquadEvaluator
from opencompass.datasets import LEvalNewsSummDataset

LEval_newssumm_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='answer',
    train_split='test',
    test_split='test'
)

LEval_newssumm_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{context}\n{question}\nTL;DR:'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512)
)

LEval_newssumm_eval_cfg = dict(
    evaluator=dict(type=RougeEvaluator), 
    pred_role='BOT'
)

LEval_newssumm_datasets = [
    dict(
        type=LEvalNewsSummDataset,
        abbr='LEval_news_summ',
        path='L4NLP/LEval',
        name='news_summ',
        reader_cfg=LEval_newssumm_reader_cfg,
        infer_cfg=LEval_newssumm_infer_cfg,
        eval_cfg=LEval_newssumm_eval_cfg)
]
