from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer, GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MedBenchDataset_v2, MedBenchEvaluator, MedBenchEvaluator_mcq
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi

# medbench_single_choice_sets = []
medbench_multiple_choices_sets = [
    'health_exam',
]
medbench_cloze_sets = ['guidance']

medbench_datasets = []
# for _name in agieval_single_choice_sets:
#     if _name in ['lsat-ar', 'lsat-lr', 'lsat-rc', 'aqua-rat']:
#         _options = ['A', 'B', 'C', 'D', 'E']
#     else:
#         _options = ['A', 'B', 'C', 'D']
#     if _name in agieval_chinese_sets:
#         _hint = '答案是：'
#     else:
#         _hint = 'The answer is '
#     agieval_infer_cfg = dict(
#         prompt_template=dict(
#             type=PromptTemplate,
#             template={
#                 label: dict(round=[
#                     dict(role='HUMAN', prompt='{question}\n{options}'),
#                     dict(role='BOT', prompt=f'{_hint}{label}')
#                 ])
#                 for label in _options
#             }),
#         retriever=dict(type=ZeroRetriever),
#         inferencer=dict(type=PPLInferencer, labels=_options))

#     agieval_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

#     agieval_datasets.append(
#         dict(
#             type=AGIEvalDataset_v2,
#             path='./data/AGIEval/data/v1/',
#             name=_name,
#             abbr='agieval-' + _name,
#             setting_name='zero-shot',
#             reader_cfg=dict(
#                 input_columns=['question', 'options'] + _options,
#                 output_column='label'),
#             infer_cfg=agieval_infer_cfg.copy(),
#             eval_cfg=agieval_eval_cfg.copy()))

for _name in medbench_multiple_choices_sets:
    _hint = '答案是： '
    medbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(role='HUMAN', prompt=f'{{question}}\n{{options}}\n{_hint}')
            ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=1024))

    medbench_eval_cfg = dict(
        evaluator=dict(type=MedBenchEvaluator_mcq),
        pred_postprocessor=dict(type=first_capital_postprocess_multi))

    medbench_datasets.append(
        dict(
            type=MedBenchDataset_v2,
            path='./data/MedBench/exam_guidance/',
            name=_name,
            abbr='medbench-' + _name,
            setting_name='zero-shot',
            reader_cfg=dict(
                input_columns=['question', 'options'], output_column='label'),
            infer_cfg=medbench_infer_cfg.copy(),
            eval_cfg=medbench_eval_cfg.copy()))

for _name in medbench_cloze_sets:
    _hint = '答案是：'
    
    medbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt=f'{{question}}{_hint}')])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=1024))

    medbench_eval_cfg = dict(evaluator=dict(type=MedBenchEvaluator))

    medbench_datasets.append(
        dict(
            type=MedBenchDataset_v2,
            path='./data/MedBench/exam_guidance/',
            name=_name,
            abbr='medbench-' + _name,
            setting_name='zero-shot',
            reader_cfg=dict(
                input_columns=['question', 'options'], output_column='label'),
            infer_cfg=medbench_infer_cfg.copy(),
            eval_cfg=medbench_eval_cfg.copy()))

for _item in medbench_datasets:
    _name = _item['name']
    _intro = {
        'health_exam':
        '以下是一道卫生资格考试选择题，请选择正确的答案。',
        'guidance':
        '请问该病人应该去哪个科室。'
    }[_name]
    _templates = _item['infer_cfg']['prompt_template']['template']

    if _item['infer_cfg']['inferencer']['type'] == PPLInferencer:
        for _label in _templates:
            _templates[_label]['round'][0][
                'prompt'] = _intro + '\n' + _templates[_label]['round'][0][
                    'prompt']
    else:
        _templates['round'][0][
            'prompt'] = _intro + '\n' + _templates['round'][0]['prompt']

del _item, _intro, _templates, _label, _name, _options, _hint, medbench_infer_cfg, medbench_eval_cfg
