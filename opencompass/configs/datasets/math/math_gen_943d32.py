from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AgentInferencer
from opencompass.datasets import MATHDataset, MATHAgentEvaluator, math_postprocess

# This config is for code interpreter
math_example = """
Example:

<HUMAN>Find the domain of the expression $\\frac{{\sqrt{{x-2}}}}{{\sqrt{{5-x}}}}$.
<ASSISTANT>{thought} The domain restrictions are determined by:

The square root in the numerator must be non-negative.
The square root in the denominator must be positive (because we can't have 0 in the denominator).
The value inside a square root (the radicand) must be non-negative. We can use `sympy` to determine the domain of the expression.
{action} PythonInterpreter
{action_input}
```python
from sympy import symbols, solveset, S, And

def solution():
    # Define the variable
    x = symbols('x')

    # Define the inequalities for the domain based on the expression
    inequality1 = x-2 >= 0  # because of the square root in the numerator
    inequality2 = 5-x > 0  # because of the square root in the denominator

    # Solve the inequalities
    domain1 = solveset(inequality1, x, domain=S.Reals)
    domain2 = solveset(inequality2, x, domain=S.Reals)

    # Find the intersection of the two domains
    final_domain = domain1.intersect(domain2)

    return final_domain

```
<SYSTEM>{response}'Interval.Ropen(2, 5)'
<ASSISTANT> {thought} Therefore, the domain of the expression is $\\boxed{{[2,5)}}$
{finish} [2,5)
"""

math_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template='{problem}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AgentInferencer, example=math_example))

math_eval_cfg = dict(
    evaluator=dict(type=MATHAgentEvaluator),
    pred_postprocessor=dict(type=math_postprocess))

math_datasets = [
    dict(
        type=MATHDataset,
        abbr='math',
        path='opencompass/math',
        reader_cfg=dict(
            input_columns=['problem'],
            output_column='solution',
        ),
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg)
]
