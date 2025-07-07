from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess_v2

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","<|end_of_text|>","\nught","\nughtuser","\nQuestion","\nnoteq","\nosterone","nosteroneuser"]

round = [
{'role': 'HUMAN', 'prompt': "Problem: Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2, \\\\\nx-5 &\\text{ if } -2 \\le x \\le 2, \\\\\n2x-b &\\text{ if } x <-2.\n\\end{array}\n\\right.\\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': "Okay, so I need to find the values of $a$ and $b$ that make this piecewise function continuous, and then add them together. Hmm, let me remember. For a piecewise function to be continuous, the left-hand limit and the right-hand limit at each boundary point must equal the function's value at that point. The function is defined in three parts: when $x > 2$, when $-2 \leq x \leq 2$, and when $x < -2$. So the critical points where the function could potentially be discontinuous are at $x = 2$ and $x = -2$. That makes sense because those are the points where the definition of the function changes.\n\nLet me start with $x = 2$. At $x = 2$, the function is defined by the middle piece, which is $f(x) = x - 5$. So first, I should check the limit as $x$ approaches $2$ from the left and from the right. Since the middle piece covers $-2 \leq x \leq 2$, approaching from the left would still be in the middle piece. Wait, actually, approaching $2$ from the left would still be in the middle piece, so the left-hand limit would just be $2 - 5 = -3$. Then approaching $2$ from the right, which is the part where $x > 2$, defined by $f(x) = ax + 3$. So the right-hand limit as $x$ approaches $2$ would be $a*(2) + 3 = 2a + 3$. For the function to be continuous at $x = 2$, these two limits must be equal to each other and also equal to $f(2)$. Since $f(2)$ is given by the middle piece, which is $-3$. Therefore, $2a + 3$ must equal $-3$.\n\nSo let me write that down:\n\nLeft-hand limit at $x = 2$: $\lim_{x \to 2^{-}} f(x)=2 - 5 = -3$.\n\nRight-hand limit at $x = 2$: $\lim_{x \to 2^{+}} f(x)=a*2 + 3 = 2a + 3$.\n\nSet them equal for continuity: $2a + 3 = -3$.\n\nSolving for $a$: subtract $3$ from both sides: $2a = -6$, then divide by $2$: $a = -3$. Okay, so $a$ is $-3$.\n\nNow, moving on to the other boundary point at $x = -2$. Let's check the continuity there. At $x = -2$, the function is defined by the middle piece: $f(x) = x - 5$. So $f(-2) = -2 -5 = -7$. Now, approaching $x = -2$ from the left ($x < -2$), the function is defined by the third piece: $f(x) = 2x - b$. So the left-hand limit as $x$ approaches $-2$ from the left would be $2*(-2) - b = -4 - b$. But wait, the left-hand limit at $x = -2$ would actually be the limit as $x$ approaches $-2$ from the left, which is $x < -2$, so using the third piece. Therefore, $\lim_{x \to -2^{-}} f(x)=2*(-2) - b = -4 - b$.\n\nApproaching $x = -2$ from the right ($x > -2$), since $x$ is approaching from the right but still in the interval $-2 \leq x \leq 2$, so we use the middle piece: $f(x) = x - 5$. Therefore, the right-hand limit is $\lim_{x \to -2^{+}} f(x)=(-2) -5 = -7$.\n\nFor continuity at $x = -2$, the left-hand limit ($-4 - b$) must equal the right-hand limit ($-7$) and also equal $f(-2) = -7$. So setting left-hand limit equal to $f(-2)$:\n\n$-4 - b = -7$.\n\nSolving for $b$: add $4$ to both sides: $-b = -3$, then multiply by $-1$: $b = 3$. Okay, so $b$ is $3$.\n\nTherefore, $a = -3$ and $b = 3$. Hence, $a + b = -3 + 3 = 0$. Let me just verify once again to make sure I didn't make any mistakes.\n\nAt $x = 2$:\n\nLeft-hand limit: $2 -5 = -3$.\n\nRight-hand limit: $a*2 +3 = (-3)*2 +3 = -6 +3 = -3$. So that's equal, good.\n\nFunction value at $x = 2$ is $-3$. So all three are equal. Good.\n\nAt $x = -2$:\n\nLeft-hand limit: $2*(-2) -b = -4 -3 = -7$.\n\nRight-hand limit: $-2 -5 = -7$.\n\nFunction value at $x = -2$ is $-7$. All three equal. Perfect.\n\nSo yes, $a = -3$ and $b = 3$. Therefore, $a + b = 0$."},
{'role': 'HUMAN', 'prompt': "Problem:\n{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
]

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=round)),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096, stopping_criteria=stop_words))

# postprocess v2
math_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2),
)

math_datasets = [
    dict(
        type=MATHDataset,
        abbr='math_r1_enhance_1shot',
        path='opencompass/math',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
