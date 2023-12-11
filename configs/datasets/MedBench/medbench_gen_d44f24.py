from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import (
    MedBenchDataset,
    MedBenchEvaluator,
    MedBenchEvaluator_Cloze,
    MedBenchEvaluator_IE,
    MedBenchEvaluator_mcq,
    MedBenchEvaluator_CMeEE,
    MedBenchEvaluator_CMeIE,
    MedBenchEvaluator_CHIP_CDEE,
    MedBenchEvaluator_CHIP_CDN,
    MedBenchEvaluator_CHIP_CTC,
    MedBenchEvaluator_NLG,
    MedBenchEvaluator_TF,
    MedBenchEvaluator_EMR,
)
from opencompass.utils.text_postprocessors import first_capital_postprocess

medbench_reader_cfg = dict(
    input_columns=['problem_input'], output_column='label')

medbench_multiple_choices_sets = ['Health_exam', 'DDx-basic', 'DDx-advanced_pre', 'DDx-advanced_final', 'SafetyBench'] # 选择题，用acc判断

medbench_qa_sets = ['Health_Counseling', 'Medicine_Counseling', 'MedDG', 'MedSpeQA', 'MedTreat', 'CMB-Clin'] # 开放式QA，有标答

medbench_cloze_sets = ['Triage'] # 限定域QA，有标答

medbench_single_choice_sets = ['Medicine_attack'] # 正确与否判断，有标答

medbench_ie_sets = ['EMR', 'CMeEE'] # 判断识别的实体是否一致，用F1评价

#, 'CMeIE', 'CHIP_CDEE', 'CHIP_CDN', 'CHIP_CTC', 'Doc_parsing', 'MRG'

medbench_datasets = []


for name in medbench_single_choice_sets:
    medbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role="HUMAN", prompt='{problem_input}')])),
        retriever=dict(type=ZeroRetriever
                       ),  # retriver 不起作用，以输入参数为准 (zero-shot / few-shot)
        inferencer=dict(type=GenInferencer))

    medbench_eval_cfg = dict(
        evaluator=dict(type=MedBenchEvaluator_TF), pred_role="BOT")

    medbench_datasets.append(
        dict(
            type=MedBenchDataset,
            path='./data/MedBench/' + name,
            name=name,
            abbr='medbench-' + name,
            setting_name='zero-shot',
            reader_cfg=medbench_reader_cfg,
            infer_cfg=medbench_infer_cfg.copy(),
            eval_cfg=medbench_eval_cfg.copy()))

for name in medbench_multiple_choices_sets:
    medbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role="HUMAN", prompt='{problem_input}')])),
        retriever=dict(type=ZeroRetriever
                       ),  # retriver 不起作用，以输入参数为准 (zero-shot / few-shot)
        inferencer=dict(type=GenInferencer))

    medbench_eval_cfg = dict(
        evaluator=dict(type=MedBenchEvaluator), pred_role="BOT")

    medbench_datasets.append(
        dict(
            type=MedBenchDataset,
            path='./data/MedBench/' + name,
            name=name,
            abbr='medbench-' + name,
            setting_name='zero-shot',
            reader_cfg=medbench_reader_cfg,
            infer_cfg=medbench_infer_cfg.copy(),
            eval_cfg=medbench_eval_cfg.copy()))

for name in medbench_qa_sets:
    medbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role="HUMAN", prompt='{problem_input}')])),
        retriever=dict(type=ZeroRetriever
                       ),  # retriver 不起作用，以输入参数为准 (zero-shot / few-shot)
        inferencer=dict(type=GenInferencer))

    medbench_eval_cfg = dict(
        evaluator=dict(type=MedBenchEvaluator_NLG), pred_role="BOT")

    medbench_datasets.append(
        dict(
            type=MedBenchDataset,
            path='./data/MedBench/' + name,
            name=name,
            abbr='medbench-' + name,
            setting_name='zero-shot',
            reader_cfg=medbench_reader_cfg,
            infer_cfg=medbench_infer_cfg.copy(),
            eval_cfg=medbench_eval_cfg.copy()))

for name in medbench_cloze_sets:
    medbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role="HUMAN", prompt='{problem_input}')])),
        retriever=dict(type=ZeroRetriever
                       ),  # retriver 不起作用，以输入参数为准 (zero-shot / few-shot)
        inferencer=dict(type=GenInferencer))

    medbench_eval_cfg = dict(
        evaluator=dict(type=MedBenchEvaluator_Cloze), pred_role="BOT")

    medbench_datasets.append(
        dict(
            type=MedBenchDataset,
            path='./data/MedBench/' + name,
            name=name,
            abbr='medbench-' + name,
            setting_name='zero-shot',
            reader_cfg=medbench_reader_cfg,
            infer_cfg=medbench_infer_cfg.copy(),
            eval_cfg=medbench_eval_cfg.copy()))

for name in medbench_ie_sets:
    medbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role="HUMAN", prompt='{problem_input}')])),
        retriever=dict(type=ZeroRetriever
                       ),  # retriver 不起作用，以输入参数为准 (zero-shot / few-shot)
        inferencer=dict(type=GenInferencer))

    medbench_eval_cfg = dict(
        evaluator=dict(type=eval('MedBenchEvaluator_'+name)), pred_role="BOT")

    medbench_datasets.append(
        dict(
            type=MedBenchDataset,
            path='./data/MedBench/' + name,
            name=name,
            abbr='medbench-' + name,
            setting_name='zero-shot',
            reader_cfg=medbench_reader_cfg,
            infer_cfg=medbench_infer_cfg.copy(),
            eval_cfg=medbench_eval_cfg.copy()))

del name, medbench_infer_cfg, medbench_eval_cfg
