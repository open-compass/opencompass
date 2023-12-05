from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchRougeEvaluator, LongBenchsamsumDataset, samsum_postprocess

LongBench_samsum_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test'
)

LongBench_samsum_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=128)
)

LongBench_samsum_eval_cfg = dict(
    evaluator=dict(type=LongBenchRougeEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=samsum_postprocess),
)

LongBench_samsum_datasets = [
    dict(
        type=LongBenchsamsumDataset,
        abbr='LongBench_samsum',
        path='THUDM/LongBench',
        name='samsum',
        reader_cfg=LongBench_samsum_reader_cfg,
        infer_cfg=LongBench_samsum_infer_cfg,
        eval_cfg=LongBench_samsum_eval_cfg)
]
