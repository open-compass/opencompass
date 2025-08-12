from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess_v2

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='Problem:\nFind the domain of the expression $\\frac{{\sqrt{{x-2}}}}{{\sqrt{{5-x}}}}$.}}\nPlease provide only the final answer, without including any intermediate reasoning steps, and put your final answer within \\boxed{}.\nSolution:'),
            dict(role='BOT', prompt='Final Answer: \\boxed{{[2,5)}}\n'),
            dict(role='HUMAN', prompt='Problem:\nIf $\det \mathbf{{A}} = 2$ and $\det \mathbf{{B}} = 12,$ then find $\det (\mathbf{{A}} \mathbf{{B}}).$\nPlease provide only the final answer, without including any intermediate reasoning steps, and put your final answer within \\boxed{}.\nSolution:'),
            dict(role='BOT', prompt='Final Answer: \\boxed{{24}}\n'),
            dict(role='HUMAN', prompt='Problem:\nTerrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?\nPlease provide only the final answer, without including any intermediate reasoning steps, and put your final answer within \\boxed{}.\nSolution:'),
            dict(role='BOT', prompt='Final Answer: \\boxed{{16}}\n'),
            dict(role='HUMAN', prompt='Problem:\nIf the system of equations: \\begin{{align*}} 6x-4y&=a,\\\\ 6y-9x &=b. \end{{align*}}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{{a}}{{b}},$ assuming $b$ is nonzero.\nPlease provide only the final answer, without including any intermediate reasoning steps, and put your final answer within \\boxed{}.\nSolution:'),
            dict(role='BOT', prompt='Final Answer: \\boxed{{-\\frac{{2}}{{3}}}}\n'),
            dict(role='HUMAN', prompt='Problem:\n{problem}\nPlease provide only the final answer, without including any intermediate reasoning steps, and put your final answer within \\boxed{}.\nSolution:\n'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

math_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2))

math_datasets = [
    dict(
        type=MATHDataset,
        abbr='math',
        path='opencompass/math',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg)
]
