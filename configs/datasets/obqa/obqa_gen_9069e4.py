from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import OBQADataset
from opencompass.utils.text_postprocessors import first_option_postprocess

_input_columns = [
    ['question_stem', 'A', 'B', 'C', 'D'],
    ['question_stem', 'A', 'B', 'C', 'D', 'fact1'],
]
_template = [
    dict(
        round=[
            dict(
                role='HUMAN',
                prompt=
                'Question: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:'
            ),
        ], ),
    dict(
        round=[
            dict(
                role='HUMAN',
                prompt=
                'Given the fact: {fact1}\nQuestion: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:',
            ),
        ], ),
]

obqa_datasets = [
    dict(
        abbr='openbookqa',
        type=OBQADataset,
        path='opencompass/openbookqa_test',
        name='main',
    ),
    dict(
        abbr='openbookqa_fact',
        type=OBQADataset,
        path='opencompass/openbookqa_fact',
        name='additional',
    ),
]

for _i in range(2):
    obqa_reader_cfg = dict(
        input_columns=_input_columns[_i], output_column='answerKey')
    obqa_infer_cfg = dict(
        prompt_template=dict(type=PromptTemplate, template=_template[_i]),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )
    obqa_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_role='BOT',
        pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
    )

    obqa_datasets[_i]['reader_cfg'] = obqa_reader_cfg
    obqa_datasets[_i]['infer_cfg'] = obqa_infer_cfg
    obqa_datasets[_i]['eval_cfg'] = obqa_eval_cfg
