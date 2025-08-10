from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AgentInferencer
from opencompass.datasets import (
    MATHDataset, MATHAgentEvaluator, math_postprocess
)

# use pal format but not perform well
math_reader_cfg = dict(input_columns=['problem'], output_column='solution')

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                # # ################################### NEW SHOT ###################################
                dict(role='HUMAN', prompt='Find the coefficient of $x^3$ when $3(x^2 - x^3+x) +3(x +2x^3- 3x^2 + 3x^5+x^3) -5(1+x-4x^3 - x^2)$ is simplifie.'),
                dict(role='BOT', prompt="""Tool:PythonInterpreter
Tool Input:```python
from sympy import symbols, simplify

def solution():
    x = symbols('x')
    expr = 3*(x**2 - x**3 + x) + 3*(x + 2*x**3 - 3*x**2 + 3*x**5 + x**3) - 5*(1 + x - 4*x**3 - x**2)
    simplified_expr = simplify(expr)

    x3_coefficient = simplified_expr.as_coefficients_dict()[x**3]
    result = x3_coefficient
    return result
```"""),
                dict(role='SYSTEM', prompt='Response:26'),
                dict(role='BOT', prompt='FinalAnswer: The final answer is $26$. I hope it is correct.'),
                dict(role='HUMAN', prompt='The surface area of a sphere with radius $r$ is $4\pi r^2$. Including the area of its circular base, what is the total surface area of a hemisphere with radius 6 cm? Express your answer in terms of $\pi$.'),
                dict(role='BOT', prompt="""Tool:PythonInterpreter
Tool Input:```python
import math

def solution():
    radius = 6

    # Surface area of the hemisphere
    hemisphere_area = 2 * math.pi * radius**2

    # Area of the circular base
    base_area = math.pi * radius**2

    # Total surface area
    total_surface_area = hemisphere_area + base_area

    # Formatting the result in LaTeX
    result = r'{}\pi'.format(total_surface_area / math.pi)
    return result
```"""),
                dict(role='SYSTEM', prompt='Response:108.0\\pi'),
                dict(role='BOT', prompt='FinalAnswer: The final answer is $108.0\pi$. I hope it is correct.'),
                dict(role='HUMAN', prompt='Monica tosses a fair 6-sided die.  If the roll is a prime number, then she wins that amount of dollars (so that, for example, if she rolls 3, then she wins 3 dollars).  If the roll is composite, she wins nothing. Otherwise, she loses 3 dollars. What is the expected value of her winnings on one die toss? Express your answer as a dollar value to the nearest cent.'),
                dict(role='BOT', prompt="""Tool:PythonInterpreter
Tool Input:```python
def solution():
    # Probabilities of each outcome
    prime_prob = 1 / 6
    composite_prob = 1 / 3
    otherwise_prob = 1 / 6

    # Expected value of each outcome
    prime_expected_value = (2 * prime_prob) + (3 * prime_prob) + (5 * prime_prob)
    composite_expected_value = 0 * composite_prob
    otherwise_expected_value = -3 * otherwise_prob

    # Total expected value
    total_expected_value = prime_expected_value + composite_expected_value + otherwise_expected_value

    # Dollar value to the nearest cent
    result = "{:.2f}".format(total_expected_value)
    return result
```"""),
                dict(role='SYSTEM', prompt='Response:1.17'),
                dict(role='BOT', prompt='FinalAnswer: The final answer is $1.17$. I hope it is correct.'),
                dict(role='HUMAN', prompt='{problem}'),
            ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AgentInferencer),
)

math_eval_cfg = dict(
    evaluator=dict(type=MATHAgentEvaluator),
    pred_postprocessor=dict(type=math_postprocess),
)

math_datasets = [
    dict(
        abbr='math-agent',
        type=MATHDataset,
        path='opencompass/math',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
