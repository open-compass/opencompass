from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.OpenFinData import OpenFinDataDataset, OpenFinDataKWEvaluator
from opencompass.utils.text_postprocessors import last_capital_postprocess

OpenFinData_datasets = []
OpenFinData_3choices_list = ['emotion_identification', 'entity_disambiguation', 'financial_facts']
OpenFinData_4choices_list = ['data_inspection', 'financial_terminology', 'metric_calculation', 'value_extraction']
OpenFinData_5choices_list = ['intent_understanding']
OpenFinData_keyword_list = ['entity_recognition']
OpenFinData_all_list = OpenFinData_3choices_list + OpenFinData_4choices_list + OpenFinData_5choices_list + OpenFinData_keyword_list

OpenFinData_eval_cfg = dict(evaluator=dict(type=AccEvaluator), pred_postprocessor=dict(type=last_capital_postprocess))
OpenFinData_KW_eval_cfg = dict(evaluator=dict(type=OpenFinDataKWEvaluator))

for _name in OpenFinData_all_list:
    if _name in OpenFinData_3choices_list:
        OpenFinData_infer_cfg = dict(
            ice_template=dict(type=PromptTemplate, template=dict(begin='</E>', round=[
                        dict(role='HUMAN', prompt=f'{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\n答案: '),
                        dict(role='BOT', prompt='{answer}')]),
                        ice_token='</E>'), retriever=dict(type=ZeroRetriever), inferencer=dict(type=GenInferencer))

        OpenFinData_datasets.append(
            dict(
                type=OpenFinDataDataset,
                path='./data/openfindata_release',
                name=_name,
                abbr='OpenFinData-' + _name,
                reader_cfg=dict(
                    input_columns=['question', 'A', 'B', 'C'],
                    output_column='answer'),
                infer_cfg=OpenFinData_infer_cfg,
                eval_cfg=OpenFinData_eval_cfg,
            ))

    if _name in OpenFinData_4choices_list:
        OpenFinData_infer_cfg = dict(
            ice_template=dict(type=PromptTemplate, template=dict(begin='</E>', round=[
                        dict(role='HUMAN', prompt=f'{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案: '),
                        dict(role='BOT', prompt='{answer}')]),
                        ice_token='</E>'), retriever=dict(type=ZeroRetriever), inferencer=dict(type=GenInferencer))

        OpenFinData_datasets.append(
            dict(
                type=OpenFinDataDataset,
                path='./data/openfindata_release',
                name=_name,
                abbr='OpenFinData-' + _name,
                reader_cfg=dict(
                    input_columns=['question', 'A', 'B', 'C', 'D'],
                    output_column='answer'),
                infer_cfg=OpenFinData_infer_cfg,
                eval_cfg=OpenFinData_eval_cfg,
            ))

    if _name in OpenFinData_5choices_list:
        OpenFinData_infer_cfg = dict(
            ice_template=dict(type=PromptTemplate, template=dict(begin='</E>', round=[
                        dict(role='HUMAN', prompt=f'{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nE. {{E}}\n答案: '),
                        dict(role='BOT', prompt='{answer}')]),
                        ice_token='</E>'), retriever=dict(type=ZeroRetriever), inferencer=dict(type=GenInferencer))

        OpenFinData_datasets.append(
            dict(
                type=OpenFinDataDataset,
                path='./data/openfindata_release',
                name=_name,
                abbr='OpenFinData-' + _name,
                reader_cfg=dict(
                    input_columns=['question', 'A', 'B', 'C', 'D', 'E'],
                    output_column='answer'),
                infer_cfg=OpenFinData_infer_cfg,
                eval_cfg=OpenFinData_eval_cfg,
            ))

    if _name in OpenFinData_keyword_list:
        OpenFinData_infer_cfg = dict(
            ice_template=dict(type=PromptTemplate, template=dict(begin='</E>', round=[
                        dict(role='HUMAN', prompt=f'{{question}}\n答案: '),
                        dict(role='BOT', prompt='{answer}')]),
                        ice_token='</E>'), retriever=dict(type=ZeroRetriever), inferencer=dict(type=GenInferencer))

        OpenFinData_datasets.append(
            dict(
                type=OpenFinDataDataset,
                path='./data/openfindata_release',
                name=_name,
                abbr='OpenFinData-' + _name,
                reader_cfg=dict(
                    input_columns=['question'],
                    output_column='answer'),
                infer_cfg=OpenFinData_infer_cfg,
                eval_cfg=OpenFinData_KW_eval_cfg,
            ))

del _name
