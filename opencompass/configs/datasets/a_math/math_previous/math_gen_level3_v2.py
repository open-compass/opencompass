from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess_v2

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","<|end_of_text|>","\nught","\nughtuser","\nQuestion","\nnoteq","\nosterone","nosteroneuser"]
round = [
{'role': 'HUMAN', 'prompt': "Question: It can be shown that for any positive integer $n,$\n\\[\\begin{pmatrix} 1 & 1 \\\\ 1 & 0 \\end{pmatrix}^n = \\begin{pmatrix} F_{n + 1} & F_n \\\\ F_n & F_{n - 1} \\end{pmatrix},\\]where $F_n$ denotes the $n$th Fibonacci number.\n\nCompute $F_{784} F_{786} - F_{785}^2.$\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': 'Since $\\begin{pmatrix} 1 & 1 \\\\ 1 & 0 \\end{pmatrix}^n = \\begin{pmatrix} F_{n + 1} & F_n \\\\ F_n & F_{n - 1} \\end{pmatrix},$\n\\[\\det \\begin{pmatrix} 1 & 1 \\\\ 1 & 0 \\end{pmatrix}^n = \\det \\begin{pmatrix} F_{n + 1} & F_n \\\\ F_n & F_{n - 1} \\end{pmatrix}.\\]Now,\n\\[\\det \\begin{pmatrix} 1 & 1 \\\\ 1 & 0 \\end{pmatrix}^n = \\left( \\det \\begin{pmatrix} 1 & 1 \\\\ 1 & 0 \\end{pmatrix} \\right)^n = (-1)^n,\\]and\n\\[\\det \\begin{pmatrix} F_{n + 1} & F_n \\\\ F_n & F_{n - 1} \\end{pmatrix} = F_{n + 1} F_{n - 1} - F_n^2,\\]so\n\\[F_{n + 1} F_{n - 1} - F_n^2 = (-1)^n.\\]In particular, taking $n = 785,$ we get $F_{784} F_{786} - F_{785}^2 = \\boxed{-1}.$\n'},
{'role': 'HUMAN', 'prompt': "Question: The sequence $a_1,$ $a_2,$ $a_3,$ $\\dots$ satisfies $a_1 = 19,$ $a_9 = 99,$ and for all $n \\ge 3,$ $a_n$ is the arithmetic mean of the first $n - 1$ terms.  Find $a_2.$\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': 'For $n \\ge 3,$ we have that\n\\[a_n = \\frac{a_1 + a_2 + \\dots + a_{n - 1}}{n - 1},\\]or\n\\[(n - 1) a_n = a_1 + a_2 + \\dots + a_{n - 1}.\\]Likewise,\n\\[n a_{n + 1} = a_1 + a_2 + \\dots + a_{n - 1} + a_n.\\]Subtracting these equations, we get\n\\[n a_{n + 1} - (n - 1) a_n = a_n,\\]so $n a_{n + 1} = n a_n.$  Then $a_{n + 1} = a_n.$\n\nThis means that the terms $a_3,$ $a_4,$ $a_5,$ $\\dots$ are all equal.  In particular, $a_3 = 99,$ so\n\\[\\frac{19 + a_2}{2} = 99.\\]We find $a_2 = \\boxed{179}.$\n'},
{'role': 'HUMAN', 'prompt': "Question: Three positive integers are each greater than $1$, have a product of $ 27000 $, and are pairwise relatively prime. What is their sum?\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': 'The prime factorization of $27000$ is $2^3\\cdot 3^3\\cdot 5^3.$ These three factors, $2^3,$ $3^3,$ and $5^3$ are pairwise relatively prime, and this is the only possible triple of positive integers satisfying the given conditions. Therefore, the answer is \\[2^3+3^3+5^3=8+27+125=\\boxed{160}.\\]\n'},
{'role': 'HUMAN', 'prompt': "Question: Given points $P(-2,7)$ and $Q(4,y)$ in a coordinate plane, for what value of $y$ is the slope of the line through $P$ and $Q$ equal to $\\frac{-3}{2}$?\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': 'The slope of a line through points $(x_1,y_1)$ and $(x_2,y_2)$  is: $$\\frac{y_2-y_1}{x_2-x_1}=\\text{slope}$$ In this case, we have: $$\\frac{y-7}{4-(-2)}=\\frac{-3}{2}$$ $$2y-14=-18$$ $$2y=-4$$ $$y=\\boxed{-2}$$\n'},
{'role': 'HUMAN', 'prompt': "Question: A triangle has side lengths of 8, 15 and 17 units. What is the area of the triangle, in square units?\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': 'We see that $8^2+15^2=64+225=289=17^2$.  So the triangle is a right triangle with legs $8$ and $15$, and thus its area is\n\n$$\\frac{8(15)}{2}=\\boxed{60}$$\n'},
{'role': 'HUMAN', 'prompt': "Question: When three standard dice are tossed, the numbers $a,b,c$ are obtained. Find the probability that $abc = 180$.\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': 'We first factor 180 into three positive integers from the set $\\{1,2,3,4,5,6\\}$. Since $180 > 5^3 = 125,$ at least one of the integers must be 6. Since $180 > 5^2\\cdot 6 = 150$, at least two integers must equal 6. Indeed, $180 = 5\\cdot6\\cdot6$ is the only such way to factor 180. Therefore, $(a,b,c) = (5,6,6), (6,5,6),(6,6,5)$ are the only possibilities for $a,b,c$. Each occurs with probability $\\left(\\frac16\\right)^3 = \\frac1{216}$, so the probability that $abc = 180$ is $3\\cdot \\frac1{216} = \\boxed{\\frac1{72}}$.\n'},
{'role': 'HUMAN', 'prompt': "Question: The radius of a cylinder is doubled and its height is tripled. If its original volume was 10 cubic feet, what is its volume now, in cubic feet?\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': "Let the cylinder's original radius and height be $r$ and $h$, respectively.  The new cylinder has volume \\[\n\\pi (2r)^2(3h)=12\\pi r^2 h,\n\\] which is 12 times larger than the original volume.  Since the original volume was 10 cubic feet, the new volume is $\\boxed{120}$ cubic feet.\n"},
{'role': 'HUMAN', 'prompt': "Question: The nonzero roots of the equation $x^2 + 6x + k = 0$ are in the ratio $2:1$. What is the value of $k$?\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': "By Vieta's formulas, the sum of the roots is $-6.$  Since they are in the ratio $2:1,$ the roots are $-4$ and $-2.$  Then $k$ is their product, namely $(-4)(-2) = \\boxed{8}.$\n"},
{'role': 'HUMAN', 'prompt': "Problem:\n{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
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
        abbr='math_level3_v2',
        path='opencompass/math',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg)
]
