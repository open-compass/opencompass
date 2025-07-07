from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess_v2

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","<|end_of_text|>","\nught","\nughtuser","\nQuestion","\nnoteq","\nosterone","nosteroneuser"]
demo = []
demo.append("Okay, so I need to figure out for which values of $b$ the number $-2$ is not in the range of the function $f(x) = x^2 + bx + 2$. Hmm, let\'s start by recalling that the range of a quadratic function depends on whether it opens upwards or downwards and the value of its vertex. Since the coefficient of $x^2$ here is 1, which is positive, the parabola opens upwards. That means the function has a minimum value at its vertex and the range is from that minimum value to infinity.\n\nSo, if $-2$ is not in the range, that means the minimum value of the function must be greater than $-2$. Therefore, I need to find the vertex of the parabola and set its y-coordinate to be greater than $-2$. The vertex of a parabola $f(x) = ax^2 + bx + c$ is at $x = \\frac{-b}{2a}$. In this case, $a = 1$, so the x-coordinate of the vertex is $x = \\frac{-b}{2}$.\n\nLet me calculate the y-coordinate of the vertex by plugging this x-value back into the function:\n$$ f\\left( \\frac{-b}{2} \\right) = \\left( \\frac{-b}{2} \\right)^2 + b \\left( \\frac{-b}{2} \\right) + 2 $$\n\nSimplifying that:\n\nFirst, square $\\frac{-b}{2}$:\n$$ \\left( \\frac{-b}{2} \\right)^2 = \\frac{b^2}{4} $$\n\nNext, multiply $b$ by $\\frac{-b}{2}$:\n$$ b \\times \\frac{-b}{2} = \\frac{-b^2}{2} $$\n\nSo, putting it all together:\n$$ \\frac{b^2}{4} + \\frac{-b^2}{2} + 2 $$\n\nCombine the $b^2$ terms:\n$$ \\frac{b^2}{4} - \\frac{2b^2}{4} = \\frac{-b^2}{4} $$\n\nTherefore, the y-coordinate of the vertex is:\n$$ \\frac{-b^2}{4} + 2 $$\n\nSo the minimum value of the function is:\n$$ 2 - \\frac{b^2}{4} $$\n\nWe want this minimum value to be greater than $-2$:\n$$ 2 - \\frac{b^2}{4} > -2 $$\n\nLet\'s solve this inequality:\n\nSubtract 2 from both sides:\n$$ -\\frac{b^2}{4} > -4 $$\n\nMultiply both sides by -4, remembering that multiplying by a negative number reverses the inequality:\n$$ b^2 < 16 $$\n\nSo, $b^2 < 16$ implies that $|b| < 4$, which does translate to $-4 < b < 4$. Therefore, the values of $b$ for which $-2$ is not in the range of $f(x)$ are those between $-4$ and $4$.\n\nBut wait, let me verify this. If $b^2 < 16$, then $|b| < 4$, which does translate to $-4 < b < 4$. So in interval notation, that\'s $(-4, 4)$.\n\nBut just to make sure I didn\'t make a mistake in the steps, let me check with a specific example. Let\'s take $b = 0$. Then the function becomes $f(x) = x^2 + 2$. The minimum value is at $x = 0$, which is 2. So the range is $[2, \infty)$, which indeed doesn\'t include -2. That\'s consistent with our result.\n\nAnother test case: if $b = 4$, then the vertex is at $x = \\frac{-4}{2} = -2$. Plugging back in,\n$$ f(-2) = (-2)^2 + 4 \\times (-2) + 2 = 4 - 8 + 2 = -2 $$\n\nSo when $b = 4$, the minimum value is exactly -2, which means -2 is in the range. Therefore, $b = 4$ is excluded, which matches our interval $(-4, 4)$. Similarly, if $b = -4$, the vertex is at $x = \\frac{-(-4)}{2} = 2$. Then,\n$$ f(2) = 4 + (-4)(2) + 2 = 4 - 8 + 2 = -2 $$\n\nSo again, -2 is in the range when $b = -4$, so those endpoints are excluded.\n\nTherefore, the correct interval for $b$ where -2 is not in the range is $(-4, 4)$.\n\nSo the answer is $\\boxed{(-4, 4)}$.")

demo.append("Okay, so I need to find the maximum possible area of a rectangle with a perimeter of 24 inches. Hmm, let's see. I remember that for a rectangle, the perimeter is calculated by the formula $P = 2*(length + width)$. In this case, the perimeter is 24 inches, so I can set up the equation: $24 = 2*(l + w)$. To simplify, divide both sides by 2, which gives me $12 = l + w$. So the sum of the length and width is 12 inches.\n\nNow, the area of a rectangle is $A = l * w$. My goal is to maximize this area given the constraint that $l + w = 12$. Hmm, how do I approach this? I think this is an optimization problem where I have to find the maximum value of the area function under a certain condition. Maybe I can express one variable in terms of the other using the perimeter equation and substitute it into the area formula. Let me try that.\n\nFrom the perimeter equation: $l + w = 12$, so let's solve for $w$: $w = 12 - l$. Then substitute this into the area formula: $A = l * (12 - l) = 12l - l²$. So the area is a quadratic function of $l$: $A(l) = -l² + 12l$.\n\nQuadratic functions have their maximum or minimum at the vertex. Since the coefficient of $l²$ is negative ($-1$), the parabola opens downward, which means the vertex is the maximum point. The vertex of a parabola given by $A(l) = al² + bl + c$ is at $l = -b/(2a)$. In this case, $a = -1$ and $b = 12$, so $l = -12/(2*(-1)) = -12/-2 = 6$. So the length that gives the maximum area is 6 inches.\n\nThen, substituting back into the equation for $w$: $w = 12 - l = 12 - 6 = 6$ inches. Wait, so both length and width are 6 inches? That means the rectangle is actually a square. Is that always the case? I remember hearing somewhere that for a given perimeter, the shape with the maximum area is a square. So that makes sense here.\n\nLet me double-check my math to be sure. The perimeter of a square with sides 6 inches would be $4*6 = 24$ inches, which matches the given perimeter. The area would be $6*6 = 36$ square inches. If I try a different rectangle, say with length 5 and width 7 (since $5 + 7 = 12$), the area would be $5*7 = 35$, which is less than 36. Similarly, if I take length 4 and width 8, the area is $4*8 = 32$, which is even smaller. So it does seem like 36 is the maximum area.\n\nAlternatively, maybe I can use calculus to confirm. Taking the area function $A(l) = -l² + 12l$, the derivative with respect to $l$ is $A’(l) = -2l + 12$. Setting the derivative equal to zero for critical points: $-2l + 12 = 0 \Rightarrow -2l = -12 \Rightarrow l = 6$. Then, the second derivative is $A''(l) = -2$, which is negative, confirming that the critical point at $l = 6$ is indeed a maximum. So calculus also gives me the same result.\n\nTherefore, the maximum area is when the rectangle is a square with sides 6 inches, and the area is 36 square inches. Let me just check one more example to make sure. If the length is 6.5 inches, then the width would be $12 - 6.5 = 5.5$ inches. The area would be $6.5*5.5 = 35.75$, which is still less than 36. Similarly, if the length is 5.5, width is 6.5, area is the same 35.75. Yep, so 36 is indeed the maximum.\n\nAnother way to think about it is that the square is the most efficient in terms of area for a given perimeter among all rectangles. So if the problem allows for the rectangle to be a square, then that's when you get the maximum area. If there were constraints that it can't be a square, then the maximum would be different, but since there are no such constraints here, the answer must be 36.\n\nAlright, so after verifying through different methods—algebraic using vertex formula, checking specific cases, calculus, and logical reasoning—it seems clear that the maximum area is 36 square inches.\n\nSo the answer is $\\boxed{36}$.")

prompt = (
    "Consider the following two reasoning process examples:\n" 
    + "Example 1 \n" + demo[0] + '\n'
    + "Example 2 \n" + demo[1] + '\n\n'
    "Please refer to the reasoning approaches demonstrated above, solve the following question by yourself:\n"
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
        abbr='math_slow_think_2shot',
        path='opencompass/math',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
