from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import RBenchDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

RBench_reader_cfg = dict(input_columns=[
    'RBench_Question_Input', 'RBench_Option_A', 'RBench_Option_B',
    'RBench_Option_C', 'RBench_Option_D', 'RBench_Option_E', 'RBench_Option_F'
],
                         output_column='target')

RBench_datasets = []

systemp_prompt_en = "Answer the following single choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEF). Think step by step before answering."

systemp_prompt_zh = '回答以下单选题。答案的最后一行应采用以下格式：“答案是$LETTER”（不带引号），其中 LETTER 是选项之一（例如 ABCDEF 之一）。回答前请逐步思考。'

RBench_infer_en_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                f'{systemp_prompt_en}\nQuestion: {{RBench_Question_Input}}\nA. {{RBench_Option_A}}\nB. {{RBench_Option_B}}\nC. {{RBench_Option_C}}\nD. {{RBench_Option_D}}\nE. {{RBench_Option_E}}\nF. {{RBench_Option_F}}\nAnswer: '
            ),
        ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

RBench_infer_zh_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                f'{systemp_prompt_zh}\n问题: {{RBench_Question_Input}}\nA. {{RBench_Option_A}}\nB. {{RBench_Option_B}}\nC. {{RBench_Option_C}}\nD. {{RBench_Option_D}}\nE. {{RBench_Option_E}}\nF. {{RBench_Option_F}}\n答案: '
            ),
        ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

RBench_eval_cfg = dict(evaluator=dict(type=AccwithDetailsEvaluator),
                       pred_postprocessor=dict(type=first_option_postprocess,
                                               options='ABCDEF'))

RBench_datasets.append(
    dict(
        abbr=f'R-Bench_en',
        type=RBenchDataset,
        path='R-Bench/R-Bench',
        name='R-Bench',
        subset='en',
        reader_cfg=RBench_reader_cfg,
        infer_cfg=RBench_infer_en_cfg,
        eval_cfg=RBench_eval_cfg,
    ))

RBench_datasets.append(
    dict(
        abbr=f'R-Bench_zh',
        type=RBenchDataset,
        path='R-Bench/R-Bench',
        name='R-Bench',
        subset='zh',
        reader_cfg=RBench_reader_cfg,
        infer_cfg=RBench_infer_zh_cfg,
        eval_cfg=RBench_eval_cfg,
    ))
