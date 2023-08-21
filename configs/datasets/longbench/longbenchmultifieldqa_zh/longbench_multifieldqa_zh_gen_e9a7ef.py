from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchF1Evaluator, LongBenchmultifieldqa_zhDataset

LongBench_multifieldqa_zh_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test'
)

LongBench_multifieldqa_zh_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=64)
)

LongBench_multifieldqa_zh_eval_cfg = dict(
    evaluator=dict(type=LongBenchF1Evaluator, language='zh'),
    pred_role='BOT'
)

LongBench_multifieldqa_zh_datasets = [
    dict(
        type=LongBenchmultifieldqa_zhDataset,
        abbr='LongBench_multifieldqa_zh',
        path='THUDM/LongBench',
        name='multifieldqa_zh',
        reader_cfg=LongBench_multifieldqa_zh_reader_cfg,
        infer_cfg=LongBench_multifieldqa_zh_infer_cfg,
        eval_cfg=LongBench_multifieldqa_zh_eval_cfg)
]
