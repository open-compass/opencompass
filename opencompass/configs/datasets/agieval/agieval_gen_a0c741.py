from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import AGIEvalDataset, AGIEvalEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess

agieval_reader_cfg = dict(
    input_columns=['problem_input'], output_column='label')

agieval_single_choice_sets = [
    'gaokao-chinese',
    'gaokao-english',
    'gaokao-geography',
    'gaokao-history',
    'gaokao-biology',
    'gaokao-chemistry',
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
    'gaokao-physics',
    'jec-qa-kd',
    'jec-qa-ca',
]
agieval_cloze_sets = ['gaokao-mathcloze', 'math']

agieval_datasets = []
for name in agieval_single_choice_sets:
    agieval_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt='{problem_input}')])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, labels=['A', 'B', 'C', 'D']))

    agieval_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(type=first_capital_postprocess))

    agieval_datasets.append(
        dict(
            type=AGIEvalDataset,
            path='opencompass/agieval',
            name=name,
            abbr='agieval-' + name,
            setting_name='zero-shot',
            reader_cfg=agieval_reader_cfg,
            infer_cfg=agieval_infer_cfg.copy(),
            eval_cfg=agieval_eval_cfg.copy()))

for name in agieval_multiple_choices_sets + agieval_cloze_sets:
    agieval_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt='{problem_input}')])),
        retriever=dict(type=ZeroRetriever
                       ),  # retriver 不起作用，以输入参数为准 (zero-shot / few-shot)
        inferencer=dict(type=GenInferencer))

    agieval_eval_cfg = dict(
        evaluator=dict(type=AGIEvalEvaluator), pred_role='BOT')

    agieval_datasets.append(
        dict(
            type=AGIEvalDataset,
            path='opencompass/agieval',
            name=name,
            abbr='agieval-' + name,
            setting_name='zero-shot',
            reader_cfg=agieval_reader_cfg,
            infer_cfg=agieval_infer_cfg.copy(),
            eval_cfg=agieval_eval_cfg.copy()))

del name, agieval_infer_cfg, agieval_eval_cfg
