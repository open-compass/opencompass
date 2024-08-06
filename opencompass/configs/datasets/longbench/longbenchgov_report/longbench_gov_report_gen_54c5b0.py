from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchRougeEvaluator, LongBenchgov_reportDataset

LongBench_gov_report_reader_cfg = dict(
    input_columns=['context'],
    output_column='answers',
    train_split='test',
    test_split='test'
)

LongBench_gov_report_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512)
)

LongBench_gov_report_eval_cfg = dict(
    evaluator=dict(type=LongBenchRougeEvaluator),
    pred_role='BOT'
)

LongBench_gov_report_datasets = [
    dict(
        type=LongBenchgov_reportDataset,
        abbr='LongBench_gov_report',
        path='THUDM/LongBench',
        name='gov_report',
        reader_cfg=LongBench_gov_report_reader_cfg,
        infer_cfg=LongBench_gov_report_infer_cfg,
        eval_cfg=LongBench_gov_report_eval_cfg)
]
