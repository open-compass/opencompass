from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import InfiniteBenchzhqaDataset, LongBenchF1Evaluator
from opencompass.utils.text_postprocessors import general_cn_postprocess

InfiniteBench_zhqa_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='answer',

)

InfiniteBench_zhqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='You are a helpful assistant.'),
            ],
            round=[
                dict(role='HUMAN', prompt='请根据以下书籍回答我的问题。\n\n{context}\n\n问题：{question}\n请尽量简短地回答。'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=40)
)

InfiniteBench_zhqa_eval_cfg = dict(
    evaluator=dict(type=LongBenchF1Evaluator, language='zh'),
    pred_role='BOT',
)

InfiniteBench_zhqa_datasets = [
    dict(
        type=InfiniteBenchzhqaDataset,
        abbr='InfiniteBench_zhqa',
        path='./data/InfiniteBench/longbook_qa_chn.jsonl',
        reader_cfg=InfiniteBench_zhqa_reader_cfg,
        infer_cfg=InfiniteBench_zhqa_infer_cfg,
        eval_cfg=InfiniteBench_zhqa_eval_cfg)
]
