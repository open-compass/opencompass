from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess_v2

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""\
Problem:
Find the domain of the expression $\\frac{{\\sqrt{{x-2}}}}{{\\sqrt{{5-x}}}}$.
Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{{[2,5)}}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.

Problem:
If $\\det \\mathbf{{A}} = 2$ and $\\det \\mathbf{{B}} = 12,$ then find $\\det (\\mathbf{{A}} \\mathbf{{B}}).$
Solution:
We have that $\\det (\\mathbf{{A}} \\mathbf{{B}}) = (\\det \\mathbf{{A}})(\\det \\mathbf{{B}}) = (2)(12) = \\boxed{{24}}.$
Final Answer: The final answer is $24$. I hope it is correct.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?
Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$: \\begin{{align*}} 30n&=480\\\\ \\Rightarrow\\qquad n&=480/30=\\boxed{{16}} \\end{{align*}}
Final Answer: The final answer is $16$. I hope it is correct.

Problem:
If the system of equations: \\begin{{align*}} 6x-4y&=a,\\\\ 6y-9x &=b. \\end{{align*}}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{{a}}{{b}},$ assuming $b$ is nonzero.
Solution:
If we multiply the first equation by $-\\frac{{3}}{{2}}$, we obtain $$6y-9x=-\\frac{{3}}{{2}}a.$$Since we also know that $6y-9x=b$, we have $$-\\frac{{3}}{{2}}a=b\\Rightarrow\\frac{{a}}{{b}}=\\boxed{{-\\frac{{2}}{{3}}}}.$$
Final Answer: The final answer is $-\\frac{{2}}{{3}}$. I hope it is correct.

Problem:
{problem}
Solution:"""
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512, stopping_criteria=['Problem']))

# postprocess v2
math_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'),
    pred_postprocessor=dict(type=math_postprocess_v2))

math_datasets = [
    dict(
        type=MATHDataset,
        abbr='math',
        path='opencompass/math',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg)
]
