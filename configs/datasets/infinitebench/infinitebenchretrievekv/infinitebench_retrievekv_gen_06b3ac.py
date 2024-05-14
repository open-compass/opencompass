from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import InfiniteBenchretrievekvDataset, InfiniteBenchretrievekvEvaluator

InfiniteBench_retrievekv_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answer',

)

InfiniteBench_retrievekv_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='You are a helpful assistant.'),
            ],
            round=[
                dict(role='HUMAN', prompt='Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=50)
)

InfiniteBench_retrievekv_eval_cfg = dict(
    evaluator=dict(type=InfiniteBenchretrievekvEvaluator),
    pred_role='BOT'
)

InfiniteBench_retrievekv_datasets = [
    dict(
        type=InfiniteBenchretrievekvDataset,
        abbr='InfiniteBench_retrievekv',
        path='./data/InfiniteBench/kv_retrieval.jsonl',
        reader_cfg=InfiniteBench_retrievekv_reader_cfg,
        infer_cfg=InfiniteBench_retrievekv_infer_cfg,
        eval_cfg=InfiniteBench_retrievekv_eval_cfg)
]
