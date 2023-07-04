from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GLMChoiceInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import AGIEvalDataset

agieval_reader_cfg = dict(
    input_columns=['problem_input'], output_column='label')

agieval_single_choice_sets = [
    'gaokao-chinese',
    'gaokao-english',
    'gaokao-geography',
    'gaokao-history',
    'gaokao-biology',
    'gaokao-chemistry',
    'gaokao-physics',
    'gaokao-mathqa',
    'logiqa-zh',
    'lsat-ar',
    'lsat-lr',
    'lsat-rc',
    'logiqa-en',
    'sat-math',
    'sat-en',
    'sat-en-without-passage',
    'aqua-rat',
]
agieval_multiple_choices_sets = [
    'jec-qa-kd',  # 数据需要额外处理
    'jec-qa-ca',  # 数据需要额外处理
]
agieval_cloze_sets = ['gaokao-mathcloze', 'math']

agieval_datasets = []
for name in agieval_single_choice_sets:
    agieval_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template={
                label: f'{{problem_input}} {label}'
                for label in ['A', 'B', 'C', 'D']
            }),
        retriever=dict(type=ZeroRetriever
                       ),  # retriver 不起作用，以输入参数为准 (zero-shot / few-shot)
        inferencer=dict(
            type=GLMChoiceInferencer, choices=['A', 'B', 'C', 'D']))

    agieval_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

    agieval_datasets.append(
        dict(
            type=AGIEvalDataset,
            path='./data/AGIEval/data/v1/',
            name=name,
            abbr='agieval-' + name,
            setting_name='zero-shot',
            reader_cfg=agieval_reader_cfg,
            infer_cfg=agieval_infer_cfg.copy(),
            eval_cfg=agieval_eval_cfg.copy()))

for name in agieval_multiple_choices_sets:
    _hint = '答案是： '
    agieval_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(role='HUMAN', prompt=f'{{question}}\n{{options}}\n{_hint}')
            ]),
            ice_token='</E>'),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type='GenInferencer'))
    agieval_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(type='first-capital-multi'))
    agieval_datasets.append(
        dict(
            type='AGIEvalDataset_v2',
            path='./data/AGIEval/data/v1/',
            name=name,
            abbr='agieval-' + name,
            setting_name='zero-shot',
            reader_cfg=agieval_reader_cfg,
            infer_cfg=agieval_infer_cfg.copy(),
            eval_cfg=agieval_eval_cfg.copy()))

for name in agieval_cloze_sets:
    agieval_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template='</E>{problem_input}',
            ice_token='</E>'),
        retriever=dict(type=ZeroRetriever
                       ),  # retriver 不起作用，以输入参数为准 (zero-shot / few-shot)
        inferencer=dict(type='GenInferencer'))

    agieval_eval_cfg = dict(evaluator=dict(type='AGIEvalEvaluator'))

    agieval_datasets.append(
        dict(
            type=AGIEvalDataset,
            path='./data/AGIEval/data/v1/',
            name=name,
            abbr='agieval-' + name,
            setting_name='zero-shot',
            reader_cfg=agieval_reader_cfg,
            infer_cfg=agieval_infer_cfg.copy(),
            eval_cfg=agieval_eval_cfg.copy()))

del name, agieval_infer_cfg, agieval_eval_cfg
