from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import InfiniteBenchenqaDataset, LongBenchF1Evaluator

InfiniteBench_enqa_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='answer',

)

InfiniteBench_enqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='You are a helpful assistant.'),
            ],
            round=[
                dict(role='HUMAN', prompt='Read the book below and answer a question.\n\n{context}\n\nQuestion: {question}\n\nBe very concise.'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=40)
)

InfiniteBench_enqa_eval_cfg = dict(
    evaluator=dict(type=LongBenchF1Evaluator),
    pred_role='BOT'
)

InfiniteBench_enqa_datasets = [
    dict(
        type=InfiniteBenchenqaDataset,
        abbr='InfiniteBench_enqa',
        path='./data/InfiniteBench/longbook_qa_eng.jsonl',
        reader_cfg=InfiniteBench_enqa_reader_cfg,
        infer_cfg=InfiniteBench_enqa_infer_cfg,
        eval_cfg=InfiniteBench_enqa_eval_cfg)
]
