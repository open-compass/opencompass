from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchRetrievalEvaluator, LongBenchpassage_retrieval_enDataset

LongBench_passage_retrieval_en_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test'
)

LongBench_passage_retrieval_en_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: '),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=32)
)

LongBench_passage_retrieval_en_eval_cfg = dict(
    evaluator=dict(type=LongBenchRetrievalEvaluator),
    pred_role='BOT'
)

LongBench_passage_retrieval_en_datasets = [
    dict(
        type=LongBenchpassage_retrieval_enDataset,
        abbr='LongBench_passage_retrieval_en',
        path='THUDM/LongBench',
        name='passage_retrieval_en',
        reader_cfg=LongBench_passage_retrieval_en_reader_cfg,
        infer_cfg=LongBench_passage_retrieval_en_infer_cfg,
        eval_cfg=LongBench_passage_retrieval_en_eval_cfg)
]
