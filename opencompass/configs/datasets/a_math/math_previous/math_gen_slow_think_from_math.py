from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess_v2

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","<|end_of_text|>","\nught","\nughtuser","\nQuestion","\nnoteq","\nosterone","nosteroneuser"]
demo = 'Okay, so I need to figure out for which values of $b$ the number $-2$ is not in the range of the function $f(x) = x^2 + bx + 2$. Hmm, let\'s start by recalling that the range of a quadratic function depends on whether it opens upwards or downwards and the value of its vertex. Since the coefficient of $x^2$ here is 1, which is positive, the parabola opens upwards. That means the function has a minimum value at its vertex and the range is from that minimum value to infinity.\n\nSo, if $-2$ is not in the range, that means the minimum value of the function must be greater than $-2$. Therefore, I need to find the vertex of the parabola and set its y-coordinate to be greater than $-2$. The vertex of a parabola $f(x) = ax^2 + bx + c$ is at $x = \\frac{-b}{2a}$. In this case, $a = 1$, so the x-coordinate of the vertex is $x = \\frac{-b}{2}$.\n\nLet me calculate the y-coordinate of the vertex by plugging this x-value back into the function:\n$$ f\\left( \\frac{-b}{2} \\right) = \\left( \\frac{-b}{2} \\right)^2 + b \\left( \\frac{-b}{2} \\right) + 2 $$\n\nSimplifying that:\n\nFirst, square $\\frac{-b}{2}$:\n$$ \\left( \\frac{-b}{2} \\right)^2 = \\frac{b^2}{4} $$\n\nNext, multiply $b$ by $\\frac{-b}{2}$:\n$$ b \\times \\frac{-b}{2} = \\frac{-b^2}{2} $$\n\nSo, putting it all together:\n$$ \\frac{b^2}{4} + \\frac{-b^2}{2} + 2 $$\n\nCombine the $b^2$ terms:\n$$ \\frac{b^2}{4} - \\frac{2b^2}{4} = \\frac{-b^2}{4} $$\n\nTherefore, the y-coordinate of the vertex is:\n$$ \\frac{-b^2}{4} + 2 $$\n\nSo the minimum value of the function is:\n$$ 2 - \\frac{b^2}{4} $$\n\nWe want this minimum value to be greater than $-2$:\n$$ 2 - \\frac{b^2}{4} > -2 $$\n\nLet\'s solve this inequality:\n\nSubtract 2 from both sides:\n$$ -\\frac{b^2}{4} > -4 $$\n\nMultiply both sides by -4, remembering that multiplying by a negative number reverses the inequality:\n$$ b^2 < 16 $$\n\nSo, $b^2 < 16$ implies that $|b| < 4$, which does translate to $-4 < b < 4$. Therefore, the values of $b$ for which $-2$ is not in the range of $f(x)$ are those between $-4$ and $4$.\n\nBut wait, let me verify this. If $b^2 < 16$, then $|b| < 4$, which does translate to $-4 < b < 4$. So in interval notation, that\'s $(-4, 4)$.\n\nBut just to make sure I didn\'t make a mistake in the steps, let me check with a specific example. Let\'s take $b = 0$. Then the function becomes $f(x) = x^2 + 2$. The minimum value is at $x = 0$, which is 2. So the range is $[2, \infty)$, which indeed doesn\'t include -2. That\'s consistent with our result.\n\nAnother test case: if $b = 4$, then the vertex is at $x = \\frac{-4}{2} = -2$. Plugging back in,\n$$ f(-2) = (-2)^2 + 4 \\times (-2) + 2 = 4 - 8 + 2 = -2 $$\n\nSo when $b = 4$, the minimum value is exactly -2, which means -2 is in the range. Therefore, $b = 4$ is excluded, which matches our interval $(-4, 4)$. Similarly, if $b = -4$, the vertex is at $x = \\frac{-(-4)}{2} = 2$. Then,\n$$ f(2) = 4 + (-4)(2) + 2 = 4 - 8 + 2 = -2 $$\n\nSo again, -2 is in the range when $b = -4$, so those endpoints are excluded.\n\nTherefore, the correct interval for $b$ where -2 is not in the range is $(-4, 4)$.\n\n\\textbf{Final Answer:}\n$$ \\boxed{(-4, 4)} $$'

prompt = (
    "Consider the following reasoning process example:\n" 
    + demo + "\n\n"
    "Please refer to the reasoning approach demonstrated above, solve the following question by yourself:\n"
    "{problem}\n\n"
    "Finally, provide your answer in the following format: 'So the answer is $\\boxed{}$'."
)

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=prompt),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096,stopping_criteria=stop_words),
)

# postprocess v2
math_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2),
)

math_datasets = [
    dict(
        type=MATHDataset,
        abbr='math_slow_think_from_math',
        path='opencompass/math',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
