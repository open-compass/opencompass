from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLOnlyInferencer
from opencompass.openicl.icl_evaluator import AveragePPLEvaluator
from opencompass.datasets import JsonlDataset

mmlu_datasets = []

mmlu_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template='{text}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLOnlyInferencer),
)

mmlu_eval_cfg = dict(evaluator=dict(type=AveragePPLEvaluator))

mmlu_reader_cfg = dict(
    input_columns=['text'],
    output_column=None,
)

mmlu_datasets.append(
    dict(
        abbr=f'mmlu-test-ppl',
        type=JsonlDataset,
        path='/mnt/petrelfs/zhoufengzhe/repos/cscripts/mock-datas/mmlu_test_content.jsonl',
        reader_cfg=mmlu_reader_cfg,
        infer_cfg=mmlu_infer_cfg,
        eval_cfg=mmlu_eval_cfg
    )
)

mmlu_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template='{rephrase}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLOnlyInferencer),
)

mmlu_eval_cfg = dict(evaluator=dict(type=AveragePPLEvaluator))

mmlu_reader_cfg = dict(
    input_columns=['rephrase'],
    output_column=None,
)

mmlu_datasets.append(
    dict(
        abbr=f'mmlu-ref-ppl',
        type=JsonlDataset,
        path='/mnt/petrelfs/zhoufengzhe/repos/cscripts/mock-datas/mmlu_test_content.jsonl',
        reader_cfg=mmlu_reader_cfg,
        infer_cfg=mmlu_infer_cfg,
        eval_cfg=mmlu_eval_cfg
    )
)
