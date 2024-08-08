from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LVEvalOPTF1Evaluator, LVEvalcmrcDataset

LVEval_cmrc_mixup_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test',
)

LVEval_cmrc_mixup_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='请根据下面给定的文章回答问题，问题和答案只与其中一篇文章有关。\n\n文章：{context}\n\n现在请基于上述文章回答下面的问题，问题和答案只与其中一篇文章有关。\n\n问题：{input}\n回答：',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=64),
)

LVEval_cmrc_mixup_eval_cfg = dict(
    evaluator=dict(type=LVEvalOPTF1Evaluator, language='zh'), pred_role='BOT'
)

DATASET_LENGTH_LEVEL = ['16k', '32k', '64k', '128k', '256k']


def get_dataset_names(dataset_name, length_levels):
    datasets = []
    for length in length_levels:
        datasets.append(f'{dataset_name}_{length}')
    return datasets


LVEval_cmrc_mixup_datasets = [
    dict(
        type=LVEvalcmrcDataset,
        abbr='LVEval_' + name_len,
        path='Infinigence/LVEval',
        name=name_len,
        reader_cfg=LVEval_cmrc_mixup_reader_cfg,
        infer_cfg=LVEval_cmrc_mixup_infer_cfg,
        eval_cfg=LVEval_cmrc_mixup_eval_cfg,
    )
    for name_len in get_dataset_names('cmrc_mixup', DATASET_LENGTH_LEVEL)
]
