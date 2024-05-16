from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MastermathDatasetv1, MastermathDatasetv1Evaluator
from opencompass.utils import first_option_postprocess

mastermath2024v1_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer')

mastermath2024v1_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{question}\n选项:\n'
                                          '(A){A}\n'
                                          '(B){B}\n'
                                          '(C){C}\n'
                                          '(D){D}\n'
                                          '你的回答格式如下: "正确答案是 (在这里插入你的答案)"'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

mastermath2024v1_eval_cfg = dict(evaluator=dict(type=MastermathDatasetv1Evaluator),
                                 pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

mastermath2024v1_datasets = [dict(
        abbr='Mastermath2024v1',
        type=MastermathDatasetv1,
        path='./data/mastermath2024v1/',
        name='kaoyan_math_1_mcq_Sheet1.csv',
        reader_cfg=mastermath2024v1_reader_cfg,
        infer_cfg=mastermath2024v1_infer_cfg,
        eval_cfg=mastermath2024v1_eval_cfg)]
