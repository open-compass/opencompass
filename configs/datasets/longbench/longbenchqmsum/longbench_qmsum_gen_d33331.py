from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchRougeEvaluator, LongBenchqmsumDataset

LongBench_qmsum_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test'
)

LongBench_qmsum_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512)
)

LongBench_qmsum_eval_cfg = dict(
    evaluator=dict(type=LongBenchRougeEvaluator),
    pred_role='BOT'
)

LongBench_qmsum_datasets = [
    dict(
        type=LongBenchqmsumDataset,
        abbr='LongBench_qmsum',
        path='THUDM/LongBench',
        name='qmsum',
        reader_cfg=LongBench_qmsum_reader_cfg,
        infer_cfg=LongBench_qmsum_infer_cfg,
        eval_cfg=LongBench_qmsum_eval_cfg)
]
