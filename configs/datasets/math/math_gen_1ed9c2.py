from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='Problem:\nFind the coefficient of $x^3$ when $3(x^2 - x^3+x) +3(x +2x^3- 3x^2 + 3x^5+x^3) -5(1+x-4x^3 - x^2)$ is simplified.\nSolution:'),
            dict(role='BOT', prompt='Combine like terms to simplify the expression. The coefficient of $x^3$ is calculated as $$(-3+2\cdot(2+1))+(-5)\cdot(-4))$ = 26$. Thus, the coefficient of $x^3$ is $\\boxed{26}$.\nFinal Answer: The final answer is $26$. I hope it is correct.\n'),
            dict(role='HUMAN', prompt='Problem:\nThe surface area of a sphere with radius $r$ is $4\pi r^2$. Including the area of its circular base, what is the total surface area of a hemisphere with radius 6 cm? Express your answer in terms of $\pi$.\nSolution:'),
            dict(role='BOT', prompt='The surface area of a hemisphere (not including the base) is half that of a sphere, so it is $2\pi r^2$. The area of the base is $\pi r^2$. Therefore, for a hemisphere with radius 6 cm, the total surface area is $2\pi (6)^2 + \pi (6)^2 = 108\pi$ square cm.\nFinal Answer: The final answer is $108\pi$ square cm. I hope it is correct.\n'),
            dict(role='HUMAN', prompt='Problem:\nMonica tosses a fair 6-sided die. If the roll is a prime number, then she wins that amount of dollars (so that, for example, if she rolls 3, then she wins 3 dollars). If the roll is composite, she wins nothing. Otherwise, she loses 3 dollars. What is the expected value of her winnings on one die toss? Express your answer as a dollar value to the nearest cent.\nSolution:'),
            dict(role='BOT', prompt='The prime numbers rolled could be 2, 3, or 5, and each has a 1/6 chance of being rolled. The composite number 4 or 6 has a 2/6 chance of being rolled, but it results in $0 win. The remaining non-prime and non-composite number is 1 , and it results in a loss of $3, with a 1/6 chance. So, the expected winnings are $(2+3+5)(1/6)+0(2/6)+(-3)(1/6) = \$1.17$.\nFinal Answer: The final answer is $\$1.17$. I hope it is correct.\n'),
            dict(role='HUMAN', prompt='Problem:\nGiven $\mathbf{a} = \\begin{pmatrix} -7 \\ 0 \\ 1 \end{pmatrix}$ and $\mathbf{b} = \\begin{pmatrix} 4 \\ 2 \\ -1 \end{pmatrix},$ find $\mathbf{a} - 3 \mathbf{b}.$\nSolution:'),
            dict(role='BOT', prompt='We find $3 \mathbf{b}$ first, which is $\\begin{pmatrix} 12 \\ 6 \\ -3 \end{pmatrix}$. Then we subtract this vector from $\mathbf{a}$. So, $\mathbf{a} - 3 \mathbf{b} = \\begin{pmatrix} -7 - 12 \\ 0 - 6 \\ 1 - (-3) \end{pmatrix} = \\begin{pmatrix} -19 \\ -6 \\ 4 \end{pmatrix}.$\nFinal Answer: The final answer is $\\begin{pmatrix} -19 \\ -6 \\ 4 \end{pmatrix}$. I hope it is correct.\n'),
            dict(role='HUMAN', prompt='Problem:\n{problem}\nSolution:\n'),
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
