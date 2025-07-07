from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess_v2

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","<|end_of_text|>","\nught","\nughtuser","\nQuestion","\nnoteq","\nosterone","nosteroneuser"]
round = [
{'role': 'HUMAN', 'prompt': "Question: What is the value of $\\dfrac{3 \\times 4}{6}?$\nLet's think step by step\nSolution:"},
{'role': 'BOT', 'prompt': 'Calculating the numerator first, $\\dfrac{3 \\times 4}{6} = \\dfrac{12}{6} = \\boxed{2}$.\n'},
{'role': 'HUMAN', 'prompt': "Question: Evaluate $3x^y + 4y^x$ when $x=2$ and $y=3$.\nLet's think step by step\nSolution:"},
{'role': 'BOT', 'prompt': 'We have $3x^y + 4y^x = 3\\cdot 2^3 + 4\\cdot 3^2 = 3\\cdot 8 + 4\\cdot 9 = 24 + 36 = \\boxed{60}$.\n'},
{'role': 'HUMAN', 'prompt': "Question: Compute $\\cos \\left( \\arcsin \\frac{5}{13} \\right).$\nLet's think step by step\nSolution:"},
{'role': 'BOT', 'prompt': 'Consider a right triangle where the opposite side is 5 and the hypotenuse is 13.\n\n[asy]\nunitsize (0.3 cm);\n\ndraw((0,0)--(12,0)--(12,5)--cycle);\n\nlabel("$12$", (6,0), S);\nlabel("$13$", (6,5/2), NW);\nlabel("$5$", (12,5/2), E);\nlabel("$\\theta$", (5,1));\n[/asy]\n\nThen $\\sin \\theta = \\frac{5}{13},$ so $\\theta = \\arcsin \\frac{5}{13}.$  By Pythagoras, the adjacent side is 12, so $\\cos \\theta = \\boxed{\\frac{12}{13}}.$\n'},
{'role': 'HUMAN', 'prompt': "Question: Cory made a complete list of the prime numbers between 1 and 25. What is the sum of the smallest prime number and the largest prime number on his list?\nLet's think step by step\nSolution:"},
{'role': 'BOT', 'prompt': 'Between 1 and 25, the smallest prime number is 2 and the largest prime number is 23. Thus the sum is $2+23=\\boxed{25}$.\n'},
{'role': 'HUMAN', 'prompt': "Question: What is the value of $\\left(1 - \\frac{1}{2}\\right)\\left(1-\\frac{1}{3}\\right)\\left(1-\\frac{1}{4}\\right)\\left(1-\\frac{1}{5}\\right) \\dotsm \\left(1-\\frac{1}{50}\\right)$? Express your answer as a common fraction.\nLet's think step by step\nSolution:"},
{'role': 'BOT', 'prompt': 'After the subtractions are performed, each fraction in the pattern has a numerator that is one less than its denominator. The product then reduces quite nicely, leaving just the frst numerator and the last denominator, as follows: $\\frac{1}{2}\\times\\frac{2}{3}\\times\\frac{3}{4}\\times \\cdots\\times\\frac{49}{50} = \\boxed{\\frac{1}{50}}$.\n'},
{'role': 'HUMAN', 'prompt': "Question: Harry, Ron and Neville are having a race on their broomsticks.  If there are no ties, in how many different possible orders can they finish?\nLet's think step by step\nSolution:"},
{'role': 'BOT', 'prompt': 'There are 3 ways to choose who finishes first.  For each possibility, there are 2 ways to choose who comes in second, and the remaining person comes in last.  That gives us $3\\cdot 2 \\cdot 1 = \\boxed{6}$ possible orders.\n'},
{'role': 'HUMAN', 'prompt': "Question: In the diagram, $BP$ and $BQ$ trisect $\\angle ABC$. $BM$ bisects $\\angle PBQ$. Find the ratio of the measure of $\\angle MBQ$ to the measure of $\\angle ABQ$.\nLet's think step by step\nSolution:"},
{'role': 'BOT', 'prompt': 'Let $\\angle MBQ = x$, so $\\angle MBP=x$ as well.  Therefore, we have $\\angle PBQ = 2x$, so $\\angle ABP = \\angle PBQ = \\angle QBC = 2x$.  Finally, we have $\\angle ABQ = \\angle ABP + \\angle PBQ = 4x$, so  \\[\\frac{\\angle MBQ}{\\angle ABQ} = \\frac{x}{4x} = \\boxed{\\frac14}.\\]\n'},
{'role': 'HUMAN', 'prompt': "Question: If $x \\cdot (x+y) = x^2 + 8$, what is the value of $xy$?\nLet's think step by step\nSolution:"},
{'role': 'BOT', 'prompt': 'Distributing the $x$ on the left-hand side gives $x^2 + xy = x^2 + 8$. Thus, $xy = \\boxed{8}$.\n'},
{'role': 'HUMAN', 'prompt': "Problem:\n{problem}\nLet's think step by step\nSolution:"},
]
# que = round[-1]
# round = [round[4].copy(), round[5].copy()] * 8
# round.append(que)
math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=round)),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096, stopping_criteria=stop_words))

math_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2))

math_datasets = [
    dict(
        type=MATHDataset,
        abbr='math_level1',
        path='opencompass/math',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg)
]
