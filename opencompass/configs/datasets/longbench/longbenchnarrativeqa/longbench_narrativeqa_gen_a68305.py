from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchF1Evaluator, LongBenchnarrativeqaDataset

LongBench_narrativeqa_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test'
)

LongBench_narrativeqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=128)
)

LongBench_narrativeqa_eval_cfg = dict(
    evaluator=dict(type=LongBenchF1Evaluator),
    pred_role='BOT'
)

LongBench_narrativeqa_datasets = [
    dict(
        type=LongBenchnarrativeqaDataset,
        abbr='LongBench_narrativeqa',
        path='THUDM/LongBench',
        name='narrativeqa',
        reader_cfg=LongBench_narrativeqa_reader_cfg,
        infer_cfg=LongBench_narrativeqa_infer_cfg,
        eval_cfg=LongBench_narrativeqa_eval_cfg)
]
