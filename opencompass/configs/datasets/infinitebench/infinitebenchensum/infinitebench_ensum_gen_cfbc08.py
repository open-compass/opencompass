from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import RougeEvaluator
from opencompass.datasets import InfiniteBenchensumDataset

InfiniteBench_ensum_reader_cfg = dict(
    input_columns=['context'],
    output_column='answer',

)

InfiniteBench_ensum_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='You are a helpful assistant.'),
            ],
            round=[
                dict(role='HUMAN', prompt='Summarize the following book.\n\n{context}'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1200)
)

InfiniteBench_ensum_eval_cfg = dict(
    evaluator=dict(type=RougeEvaluator),
    pred_role='BOT'
)

InfiniteBench_ensum_datasets = [
    dict(
        type=InfiniteBenchensumDataset,
        abbr='InfiniteBench_ensum',
        path='./data/InfiniteBench/longbook_sum_eng.jsonl',
        reader_cfg=InfiniteBench_ensum_reader_cfg,
        infer_cfg=InfiniteBench_ensum_infer_cfg,
        eval_cfg=InfiniteBench_ensum_eval_cfg)
]
