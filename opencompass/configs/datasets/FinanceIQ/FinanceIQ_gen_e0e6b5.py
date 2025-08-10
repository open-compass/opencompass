from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import FinanceIQDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess

financeIQ_subject_mapping_en = {
    'certified_public_accountant': '注册会计师（CPA）',
    'banking_qualification': '银行从业资格',
    'securities_qualification': '证券从业资格',
    'fund_qualification': '基金从业资格',
    'insurance_qualification': '保险从业资格CICE',
    'economic_analyst': '经济师',
    'taxation_practitioner': '税务师',
    'futures_qualification': '期货从业资格',
    'certified_fin_planner': '理财规划师',
    'actuary_fin_math': '精算师-金融数学',
}

financeIQ_subject_mapping = {
    '注册会计师（CPA）': '注册会计师（CPA）',
    '银行从业资格': '银行从业资格',
    '证券从业资格': '证券从业资格',
    '基金从业资格': '基金从业资格',
    '保险从业资格CICE': '保险从业资格CICE',
    '经济师': '经济师',
    '税务师': '税务师',
    '期货从业资格': '期货从业资格',
    '理财规划师': '理财规划师',
    '精算师-金融数学': '精算师-金融数学',
}

financeIQ_all_sets = list(financeIQ_subject_mapping.keys())

financeIQ_datasets = []
for _name in financeIQ_all_sets:
    _ch_name = financeIQ_subject_mapping[_name]
    financeIQ_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(
                        role='HUMAN',
                        prompt=
                        f'以下是关于{_ch_name}的单项选择题，请直接给出正确答案的选项。\n题目：{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}'
                    ),
                    dict(role='BOT', prompt='答案是: {answer}'),
                ]),
            ice_token='</E>',
        ),
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
        inferencer=dict(type=GenInferencer),
    )

    financeIQ_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(type=first_capital_postprocess))

    financeIQ_datasets.append(
        dict(
            type=FinanceIQDataset,
            path='./data/FinanceIQ/',
            name=_name,
            abbr=f'FinanceIQ-{_name}',
            reader_cfg=dict(
                input_columns=['question', 'A', 'B', 'C', 'D'],
                output_column='answer',
                train_split='dev',
                test_split='test'),
            infer_cfg=financeIQ_infer_cfg,
            eval_cfg=financeIQ_eval_cfg,
        ))

del _name, _ch_name
