from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLOnlyInferencer
from opencompass.openicl.icl_evaluator import AveragePPLEvaluator
from opencompass.datasets import SanitizedMBPPDataset, JsonlDataset

mbpp_datasets = []

mbpp_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template='{text}\n{code}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLOnlyInferencer),
)

mbpp_eval_cfg = dict(evaluator=dict(type=AveragePPLEvaluator))

for split in ['train', 'test']:
    mbpp_reader_cfg = dict(
        input_columns=['text', 'code'],
        output_column=None,
        train_split=split,
        test_split=split,
    )
    mbpp_datasets.append(
        dict(
            abbr=f'mbpp-{split}-ppl',
            type=SanitizedMBPPDataset,
            path='opencompass/sanitized_mbpp',
            reader_cfg=mbpp_reader_cfg,
            infer_cfg=mbpp_infer_cfg,
            eval_cfg=mbpp_eval_cfg)
    )


mbpp_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template='{text}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLOnlyInferencer),
)

mbpp_eval_cfg = dict(evaluator=dict(type=AveragePPLEvaluator))

mbpp_reader_cfg = dict(
    input_columns=['text'],
    output_column=None,
)

mbpp_datasets.append(
    dict(
        abbr=f'mbpp-ref-ppl',
        type=JsonlDataset,
        path='/mnt/petrelfs/zhoufengzhe/repos/cscripts/mock-datas/mock_mbpp_20240113.jsonl',
        reader_cfg=mbpp_reader_cfg,
        infer_cfg=mbpp_infer_cfg,
        eval_cfg=mbpp_eval_cfg
    )
)
