from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HungarianExamMathDataset

hungarianmath_reader_cfg = dict(input_columns=['question'], output_column=None)

template = """Problem:
Find the domain of the expression $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.

Solution:
To determine the domain, we must ensure that:
1. The expressions inside each square root are non-negative.
2. The denominator is not equal to zero.

For the numerator, $x-2 \ge 0$ gives $x \ge 2$.

For the denominator, $5-x \ge 0$ gives $x \le 5$. And since the denominator cannot be zero, $5-x > 0$ which further narrows it to $x < 5$.

Combining these results, the domain of the expression is $[2,5)$.

Final Answer: The final answer is $[2,5)$.

Problem:
If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12$, then find $\det (\mathbf{A} \mathbf{B})$.

Solution:
Using the property of determinants, we can say that:
$\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B})$.
Plugging in the given values:
$\det (\mathbf{A} \mathbf{B}) = 2 \times 12 = 24$.

Final Answer: The final answer is $24$.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
First, calculate the total weight Terrell lifts with the 20-pound weights:
$2 \times 12 \times 20 = 480$ pounds.
If he uses 15-pound weights and lifts them $n$ times:
$2 \times 15 \times n = 30n$ pounds.
To find $n$, set these two equal:
\begin{align*}
30n &= 480 \\
n &= \frac{480}{30} \\
n &= 16
\end{align*}

Final Answer: The final answer is $16$.

Problem:
If the system of equations
\begin{align*}
6x-4y &= a, \\
6y-9x &= b.
\end{align*}
has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\frac{a}{b}$, assuming $b$ is nonzero.

Solution:
Multiply the first equation by $-\frac{3}{2}$ to obtain:
$6y-9x = -\frac{3}{2}a$.
Since we also know that $6y-9x = b$, equating them gives:
$-\frac{3}{2}a = b$ which implies $\frac{a}{b} = -\frac{2}{3}$.

Final Answer: The final answer is $-\frac{2}{3}$."""

hungarianmath_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=template+'\n\nProblem:\n{question}\n\nSolution:\n'),
            ],
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024))

# Attention: this math dataset needs human to evaluate the generated answer, so the AccEvaluator is just a placeholder.
hungarianmath_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

hungarianmath_datasets = [
    dict(
        abbr='HungarianExamMath',
        type=HungarianExamMathDataset,
        path='./data/HungarianExamMath/test.csv',
        reader_cfg=hungarianmath_reader_cfg,
        infer_cfg=hungarianmath_infer_cfg,
        eval_cfg=hungarianmath_eval_cfg)
]
