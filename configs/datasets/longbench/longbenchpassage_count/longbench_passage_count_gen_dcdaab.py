from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchCountEvaluator, LongBenchpassage_countDataset

LongBench_passage_count_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test'
)

LongBench_passage_count_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: '),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=32)
)

LongBench_passage_count_eval_cfg = dict(
    evaluator=dict(type=LongBenchCountEvaluator),
    pred_role='BOT'
)

LongBench_passage_count_datasets = [
    dict(
        type=LongBenchpassage_countDataset,
        abbr='LongBench_passage_count',
        path='THUDM/LongBench',
        name='passage_count',
        reader_cfg=LongBench_passage_count_reader_cfg,
        infer_cfg=LongBench_passage_count_infer_cfg,
        eval_cfg=LongBench_passage_count_eval_cfg)
]
