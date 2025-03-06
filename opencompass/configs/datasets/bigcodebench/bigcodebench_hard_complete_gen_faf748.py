from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (BigCodeBenchDataset, BigCodeBenchEvaluator)

bigcodebench_hard_reader_cfg = dict(
    input_columns=['complete_prompt'],
    output_column='test',
)

bigcodebench_hard_infer_cfg = dict(prompt_template=dict(
    type=PromptTemplate,
    template=dict(
        begin=[dict(role='system', fallback_role='HUMAN', prompt='')],
        round=[
            dict(role='HUMAN', prompt='{complete_prompt}'),
        ])),
                                   retriever=dict(type=ZeroRetriever),
                                   inferencer=dict(type=GenInferencer,
                                                   max_out_len=1024))

bigcodebench_hard_eval_cfg = dict(
    evaluator=dict(
        type=BigCodeBenchEvaluator,
        release_version='v0.1.2',
        eval_type='complete',
        # remote_execute_api='https://bigcode-bigcodebench-evaluator.hf.space/',
        remote_execute_api=
        'https://opencompass-opencompass-bigcodebench-evaluator.hf.space',  # noqa: E501
        dataset_version='hard',
    ),
    pred_role='BOT',
)

bigcodebench_hard_complete_datasets = [
    dict(
        abbr='bigcodebench_hard_complete',
        type=BigCodeBenchDataset,
        path='opencompass/bigcodebench',
        reader_cfg=bigcodebench_hard_reader_cfg,
        infer_cfg=bigcodebench_hard_infer_cfg,
        eval_cfg=bigcodebench_hard_eval_cfg,
        release_version='v0.1.2',
        dataset_version='hard',
    )
]
