from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLOnlyInferencer
from opencompass.openicl.icl_evaluator import AveragePPLEvaluator
from opencompass.datasets import JsonlDataset

ceval_datasets = []

ceval_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template='{text}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLOnlyInferencer),
)

ceval_eval_cfg = dict(evaluator=dict(type=AveragePPLEvaluator))

ceval_reader_cfg = dict(
    input_columns=['text'],
    output_column=None,
)

ceval_datasets.append(
    dict(
        abbr=f'ceval-val-ppl',
        type=JsonlDataset,
        path='/mnt/petrelfs/zhoufengzhe/repos/cscripts/mock-datas/ceval_val_content.jsonl',
        reader_cfg=ceval_reader_cfg,
        infer_cfg=ceval_infer_cfg,
        eval_cfg=ceval_eval_cfg
    )
)

ceval_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template='{rephrase}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLOnlyInferencer),
)

ceval_eval_cfg = dict(evaluator=dict(type=AveragePPLEvaluator))

ceval_reader_cfg = dict(
    input_columns=['rephrase'],
    output_column=None,
)

ceval_datasets.append(
    dict(
        abbr=f'ceval-ref-ppl',
        type=JsonlDataset,
        path='/mnt/petrelfs/zhoufengzhe/repos/cscripts/mock-datas/ceval_val_content.jsonl',
        reader_cfg=ceval_reader_cfg,
        infer_cfg=ceval_infer_cfg,
        eval_cfg=ceval_eval_cfg
    )
)
