from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LVEvalOPTF1Evaluator, LVEvallooglecrDataset

LVEval_loogle_CR_mixup_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test',
)

LVEval_loogle_CR_mixup_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='Please answer the following question based on the given passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\nPlease answer the following question based on the above passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nQuestion: {input}\nAnswer:',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=64),
)

LVEval_loogle_CR_mixup_eval_cfg = dict(
    evaluator=dict(type=LVEvalOPTF1Evaluator, language='en'), pred_role='BOT'
)

DATASET_LENGTH_LEVEL = ['16k', '32k', '64k', '128k', '256k']


def get_dataset_names(dataset_name, length_levels):
    datasets = []
    for length in length_levels:
        datasets.append(f'{dataset_name}_{length}')
    return datasets


LVEval_loogle_CR_mixup_datasets = [
    dict(
        type=LVEvallooglecrDataset,
        abbr='LVEval_' + name_len,
        path='Infinigence/LVEval',
        name=name_len,
        reader_cfg=LVEval_loogle_CR_mixup_reader_cfg,
        infer_cfg=LVEval_loogle_CR_mixup_infer_cfg,
        eval_cfg=LVEval_loogle_CR_mixup_eval_cfg,
    )
    for name_len in get_dataset_names('loogle_CR_mixup', DATASET_LENGTH_LEVEL)
]
