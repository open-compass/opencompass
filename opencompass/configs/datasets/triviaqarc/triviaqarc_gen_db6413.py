from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TriviaQArcDataset, TriviaQAEvaluator

triviaqarc_reader_cfg = dict(
    input_columns=['question', 'evidence'],
    output_column='answer',
    train_split='dev',
    test_split='dev')

triviaqarc_infer_cfg = dict(
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

triviaqarc_eval_cfg = dict(
    evaluator=dict(type=TriviaQAEvaluator), pred_role='BOT')

triviaqarc_datasets = [
    dict(
        type=TriviaQArcDataset,
        abbr='triviaqarc',
        path='./data/triviaqa-rc/',
        reader_cfg=triviaqarc_reader_cfg,
        infer_cfg=triviaqarc_infer_cfg,
        eval_cfg=triviaqarc_eval_cfg)
]
