from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import NarrativeQADataset, TriviaQAEvaluator

narrativeqa_reader_cfg = dict(
    input_columns=['question', 'evidence'],
    output_column='answer',
    train_split='valid',
    test_split='valid')

narrativeqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{evidence}\nAnswer these questions:\nQ: {question}?A:'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer, max_out_len=50, max_seq_len=8192, batch_size=4))

narrativeqa_eval_cfg = dict(
    evaluator=dict(type=TriviaQAEvaluator), pred_role='BOT')

narrativeqa_datasets = [
    dict(
        type=NarrativeQADataset,
        abbr='NarrativeQA',
        path='./data/narrativeqa/',
        reader_cfg=narrativeqa_reader_cfg,
        infer_cfg=narrativeqa_infer_cfg,
        eval_cfg=narrativeqa_eval_cfg)
]
