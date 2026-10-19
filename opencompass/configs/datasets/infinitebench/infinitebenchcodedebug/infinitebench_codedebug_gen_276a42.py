from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess
from opencompass.datasets import InfiniteBenchcodedebugDataset

InfiniteBench_codedebug_reader_cfg = dict(
    input_columns=['context', 'question', 'option_A', 'option_B', 'option_C', 'option_D'],
    output_column='answer',

)

InfiniteBench_codedebug_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='You are a helpful assistant.'),
            ],
            round=[
                dict(role='HUMAN', prompt='There is ONLY ONE function in the large project that is deliberately made to include an obvious error. Please find the function that contains the most obvious errors. I will give you four options to narrow your scope. You can inspect the options and think. Eventually, tell me the answer using one single letter (A, B, C, or D).\n\n{context}\n\nWhich funtion has deliberate error?\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nYou should first find the functions in the options. Repeat their content, inspect through code, and at last give me your answer for the function that has the deliberate and obvious error in A, B, C, or D.'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=5)
)

InfiniteBench_codedebug_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
    pred_role='BOT'
)

InfiniteBench_codedebug_datasets = [
    dict(
        type=InfiniteBenchcodedebugDataset,
        abbr='InfiniteBench_codedebug',
        path='./data/InfiniteBench/code_debug.jsonl',
        reader_cfg=InfiniteBench_codedebug_reader_cfg,
        infer_cfg=InfiniteBench_codedebug_infer_cfg,
        eval_cfg=InfiniteBench_codedebug_eval_cfg)
]
