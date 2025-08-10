from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import InfiniteBenchmathfindDataset
from opencompass.datasets.infinitebench.utils import InfiniteBench_first_number_postprocess

InfiniteBench_mathfind_reader_cfg = dict(
    input_columns=['prefix', 'context', 'question'],
    output_column='answer',

)

InfiniteBench_mathfind_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='You are a helpful assistant.'),
            ],
            round=[
                dict(role='HUMAN', prompt='{prefix}\n\n{context}\n\n{input}'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=3)
)

InfiniteBench_mathfind_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=InfiniteBench_first_number_postprocess),
    pred_role='BOT'
)

InfiniteBench_mathfind_datasets = [
    dict(
        type=InfiniteBenchmathfindDataset,
        abbr='InfiniteBench_mathfind',
        path='./data/InfiniteBench/math_find.jsonl',
        reader_cfg=InfiniteBench_mathfind_reader_cfg,
        infer_cfg=InfiniteBench_mathfind_infer_cfg,
        eval_cfg=InfiniteBench_mathfind_eval_cfg)
]
