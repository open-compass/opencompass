from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import BuySideFinBenchDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

# Subject mapping: file_name -> (Chinese display name, English display name)
BuySideFinBench_subject_mapping = {
    'three_statements_zh': ('三表联动', 'Three-Statement Linkage'),
    'three_statements_en': ('Three-Statement Linkage', 'Three-Statement Linkage'),
    'dcf_valuation_zh': ('DCF估值', 'DCF Valuation'),
    'dcf_valuation_en': ('DCF Valuation', 'DCF Valuation'),
    'comps_analysis_zh': ('可比公司分析', 'Comparable Company Analysis'),
    'comps_analysis_en': ('Comparable Company Analysis',
                          'Comparable Company Analysis'),
    'financial_ratios_zh': ('财务比率', 'Financial Ratios'),
    'financial_ratios_en': ('Financial Ratios', 'Financial Ratios'),
    'accounting_standards_zh': ('会计准则', 'Accounting Standards'),
    'accounting_standards_en': ('Accounting Standards',
                                'Accounting Standards'),
    'sensitivity_scenario_zh': ('敏感性与情景分析',
                                'Sensitivity & Scenario Analysis'),
    'sensitivity_scenario_en': ('Sensitivity & Scenario Analysis',
                                'Sensitivity & Scenario Analysis'),
}

BuySideFinBench_all_sets = list(BuySideFinBench_subject_mapping.keys())

BuySideFinBench_datasets = []
for _name in BuySideFinBench_all_sets:
    _ch_name, _en_name = BuySideFinBench_subject_mapping[_name]
    _is_zh = _name.endswith('_zh')

    if _is_zh:
        _prompt = (f'以下是关于{_ch_name}的单项选择题，请直接给出正确答案的选项。\n'
                   f'题目：{{question}}\nA. {{A}}\nB. {{B}}\n'
                   f'C. {{C}}\nD. {{D}}')
        _answer_prefix = '答案是: {answer}'
    else:
        _prompt = (
            f'The following is a multiple-choice question about {_en_name}. '
            f'Please provide the correct answer option directly.\n'
            f'Question: {{question}}\nA. {{A}}\nB. {{B}}\n'
            f'C. {{C}}\nD. {{D}}')
        _answer_prefix = 'The answer is: {answer}'

    _infer_cfg = dict(
        ice_template=dict(
            type=RawPromptTemplate,
            messages=[
                dict(role='user', content=_prompt),
                dict(role='assistant', content=_answer_prefix),
            ],
        ),
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                '</E>',
                dict(role='user', content=_prompt),
            ],
        ),
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
        inferencer=dict(type=GenInferencer),
    )

    _eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

    BuySideFinBench_datasets.append(
        dict(
            type=BuySideFinBenchDataset,
            path='cindy90/BuySideFinBench',
            name=_name,
            abbr=f'BuySideFinBench-{_name}',
            reader_cfg=dict(
                input_columns=['question', 'A', 'B', 'C', 'D'],
                output_column='answer',
                train_split='dev',
                test_split='test'),
            infer_cfg=_infer_cfg,
            eval_cfg=_eval_cfg,
        ))

del _name, _ch_name, _en_name, _is_zh, _prompt, _answer_prefix
