from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset


_hint = 'The following are semantic matching questions. \n' \
    'Please determine whether the following two sentences are semantically equivalent: ' \
    '0 means not equivalent, 1 means equivalent.\n'
MRPC_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template='Sentence one: {sentence1}\nSentence two: {sentence2}\nResult: {label}',
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template={
            answer:
            f'{_hint}</E>Sentence one: {{sentence1}}\nSentence two: {{sentence2}}\nResult: {answer}'
            for answer in [0, 1]
        },
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
    inferencer=dict(type=PPLInferencer))

MRPC_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )


MRPC_datasets = []
for _split in ['validation', 'test']:

    MRPC_reader_cfg = dict(
        input_columns=['sentence1', 'sentence2'],
        output_column='label',
        train_split='train',
        test_split=_split
    )

    MRPC_datasets.append(
        dict(
            abbr=f'MRPC-{_split}',
            type=HFDataset,
            path='glue',
            name='mrpc',
            reader_cfg=MRPC_reader_cfg,
            infer_cfg=MRPC_infer_cfg,
            eval_cfg=MRPC_eval_cfg
        )
    )
