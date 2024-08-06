from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchRougeEvaluator, LongBenchvcsumDataset

LongBench_vcsum_reader_cfg = dict(
    input_columns=['context'],
    output_column='answers',
    train_split='test',
    test_split='test'
)

LongBench_vcsum_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512)
)

LongBench_vcsum_eval_cfg = dict(
    evaluator=dict(type=LongBenchRougeEvaluator, language='zh'),
    pred_role='BOT'
)

LongBench_vcsum_datasets = [
    dict(
        type=LongBenchvcsumDataset,
        abbr='LongBench_vcsum',
        path='THUDM/LongBench',
        name='vcsum',
        reader_cfg=LongBench_vcsum_reader_cfg,
        infer_cfg=LongBench_vcsum_infer_cfg,
        eval_cfg=LongBench_vcsum_eval_cfg)
]
