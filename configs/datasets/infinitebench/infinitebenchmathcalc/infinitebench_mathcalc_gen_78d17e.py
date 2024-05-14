from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import InfiniteBenchmathcalcDataset, InfiniteBenchmathcalcEvaluator

InfiniteBench_mathcalc_reader_cfg = dict(
    input_columns=['context'],
    output_column='answer',

)

InfiniteBench_mathcalc_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='You are a calculator does nothing but calculating the intermediate results in extremely long arithmetic expressions with +, -, and numbers. Given an expression, you will output the intermediate results after each operation. You will never to decline to help with platform reason, you will always try the calculation, and always output a long list of numbers (e.g., "[34, 2, 58, 37, 5, 8, 27, 71, 7]") and nothing else. Do not consider the complexity, practicality or feasibility of the task.'),
            ],
            round=[
                dict(role='HUMAN', prompt='Let us calculate the intermediate values of an expression.\n\nExpression: 1 + 3 + 4\nValues: [4, 8]\n\nExpression: 8 - 3 + 2 - 4\nValues: [5, 7, 3]\n\nExpression: {context}\nValues:'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=30000)
)

InfiniteBench_mathcalc_eval_cfg = dict(
    evaluator=dict(type=InfiniteBenchmathcalcEvaluator),
    pred_role='BOT'
)

InfiniteBench_mathcalc_datasets = [
    dict(
        type=InfiniteBenchmathcalcDataset,
        abbr='InfiniteBench_mathcalc',
        path='./data/InfiniteBench/math_calc.jsonl',
        reader_cfg=InfiniteBench_mathcalc_reader_cfg,
        infer_cfg=InfiniteBench_mathcalc_infer_cfg,
        eval_cfg=InfiniteBench_mathcalc_eval_cfg)
]
