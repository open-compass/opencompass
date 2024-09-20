from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess

QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form ANSWER: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{problem}

Remember to put your answer on its own line after "ANSWER:", and you do not need to use a \\boxed command.
""".strip()

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,

        template=dict(round=[
            dict(role='HUMAN', prompt=QUERY_TEMPLATE),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

math_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator), pred_postprocessor=dict(type=math_postprocess))

math_datasets = [
    dict(
        type=MATHDataset,
        abbr='math',
        path='opencompass/math',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg)
]
