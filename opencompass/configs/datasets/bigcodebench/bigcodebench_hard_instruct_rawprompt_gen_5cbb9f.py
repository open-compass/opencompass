from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (BigCodeBenchDataset, BigCodeBenchEvaluator)

bigcodebench_hard_reader_cfg = dict(
    input_columns=['instruct_prompt'],
    output_column='test',
)

bigcodebench_hard_infer_cfg = dict(prompt_template=dict(
    type=RawPromptTemplate,
    messages=[
        {'role': 'user', 'content': '{instruct_prompt}'},
    ]),
       retriever=dict(type=ZeroRetriever),
       inferencer=dict(type=GenInferencer)
)

bigcodebench_hard_eval_cfg = dict(
    evaluator=dict(
        type=BigCodeBenchEvaluator,
        release_version='v0.1.2',
        eval_type='instruct',
        # remote_execute_api='https://bigcode-bigcodebench-evaluator.hf.space/',
        remote_execute_api=
        'https://opencompass-opencompass-bigcodebench-evaluator.hf.space',  # noqa: E501
        dataset_version='hard',
    ),
    pred_role='BOT',
)

bigcodebench_hard_instruct_datasets = [
    dict(
        abbr='bigcodebench_hard_instruct',
        type=BigCodeBenchDataset,
        path='opencompass/bigcodebench',
        reader_cfg=bigcodebench_hard_reader_cfg,
        infer_cfg=bigcodebench_hard_infer_cfg,
        eval_cfg=bigcodebench_hard_eval_cfg,
        release_version='v0.1.2',
        dataset_version='hard',
    )
]
