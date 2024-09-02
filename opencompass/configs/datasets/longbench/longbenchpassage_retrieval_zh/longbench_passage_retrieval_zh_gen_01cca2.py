from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchRetrievalEvaluator, LongBenchpassage_retrieval_zhDataset

LongBench_passage_retrieval_zh_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test'
)

LongBench_passage_retrieval_zh_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=32)
)

LongBench_passage_retrieval_zh_eval_cfg = dict(
    evaluator=dict(type=LongBenchRetrievalEvaluator, language='zh'),
    pred_role='BOT'
)

LongBench_passage_retrieval_zh_datasets = [
    dict(
        type=LongBenchpassage_retrieval_zhDataset,
        abbr='LongBench_passage_retrieval_zh',
        path='THUDM/LongBench',
        name='passage_retrieval_zh',
        reader_cfg=LongBench_passage_retrieval_zh_reader_cfg,
        infer_cfg=LongBench_passage_retrieval_zh_infer_cfg,
        eval_cfg=LongBench_passage_retrieval_zh_eval_cfg)
]
