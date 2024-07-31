from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LVEvalF1Evaluator, LVEvalfactrecallenDataset

LVEval_factrecall_en_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test',
)

LVEval_factrecall_en_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='Please answer the following questions based on the given article.\n\nArticle: {context}\n\nPlease answer the following questions based on the above article.\n\nQuestion: {input}\nAnswer:',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=16),
)

LVEval_factrecall_en_eval_cfg = dict(
    evaluator=dict(type=LVEvalF1Evaluator, language='en'), pred_role='BOT'
)

DATASET_LENGTH_LEVEL = ['16k', '32k', '64k', '128k', '256k']


def get_dataset_names(dataset_name, length_levels):
    datasets = []
    for length in length_levels:
        datasets.append(f'{dataset_name}_{length}')
    return datasets


LVEval_factrecall_en_datasets = [
    dict(
        type=LVEvalfactrecallenDataset,
        abbr='LVEval_' + name_len,
        path='Infinigence/LVEval',
        name=name_len,
        reader_cfg=LVEval_factrecall_en_reader_cfg,
        infer_cfg=LVEval_factrecall_en_infer_cfg,
        eval_cfg=LVEval_factrecall_en_eval_cfg,
    )
    for name_len in get_dataset_names('factrecall_en', DATASET_LENGTH_LEVEL)
]
