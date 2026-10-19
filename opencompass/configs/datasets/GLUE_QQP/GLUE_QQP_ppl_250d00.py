from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset


_hint = 'The following are semantic matching questions. \n' \
    'Please determine whether the following two sentences are semantically duplicate: ' \
    '0 means not duplicate, 1 means duplicate.\n'
QQP_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template='Sentence one: {question1}\nSentence two: {question2}\nResult: {label}',
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template={
            answer:
            f'{_hint}</E>Sentence one: {{question1}}\nSentence two: {{question2}}\nResult: {answer}'
            for answer in [0, 1]
        },
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
    inferencer=dict(type=PPLInferencer))

QQP_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )


QQP_datasets = []
for _split in ['validation', 'test']:

    QQP_reader_cfg = dict(
        input_columns=['question1', 'question2'],
        output_column='label',
        train_split='train',
        test_split=_split
    )

    QQP_datasets.append(
        dict(
            abbr=f'QQP-{_split}',
            type=HFDataset,
            path='glue',
            name='qqp',
            reader_cfg=QQP_reader_cfg,
            infer_cfg=QQP_infer_cfg,
            eval_cfg=QQP_eval_cfg
        )
    )
