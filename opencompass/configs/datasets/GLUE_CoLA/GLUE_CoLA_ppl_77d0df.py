from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset


_hint = 'The following are text classification questions. \n' \
    'Please determine whether the following sentence is linguistically acceptable: ' \
    '0 means unacceptable, 1 means acceptable.\n'

CoLA_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template='Sentence: {sentence}\nResult: {label}',
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template={
            answer:
            f'{_hint}</E>Sentence: {{sentence}}\nResult: {answer}'
            for answer in [0, 1]
        },
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[17, 18, 19, 20, 21]),
    inferencer=dict(type=PPLInferencer))

CoLA_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )

CoLA_datasets = []
for _split in ['validation']:

    CoLA_reader_cfg = dict(
        input_columns=['sentence'],
        output_column='label',
        test_split=_split
    )

    CoLA_datasets.append(
        dict(
            abbr=f'CoLA-{_split}',
            type=HFDataset,
            path='glue',
            name='cola',
            reader_cfg=CoLA_reader_cfg,
            infer_cfg=CoLA_infer_cfg,
            eval_cfg=CoLA_eval_cfg
        )
    )
