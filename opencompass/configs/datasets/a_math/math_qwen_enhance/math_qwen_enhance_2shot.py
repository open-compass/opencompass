from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess_v2

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","<|end_of_text|>","\nught","\nughtuser","\nQuestion","\nnoteq","\nosterone","nosteroneuser"]

round = [
{'role': 'HUMAN', 'prompt': "Problem: If $\\mathbf{a},$ $\\mathbf{b},$ and $\\mathbf{c}$ are unit vectors, then find the largest possible value of\n\\[\\|\\mathbf{a} - \\mathbf{b}\\|^2 + \\|\\mathbf{a} - \\mathbf{c}\\|^2 + \\|\\mathbf{b} - \\mathbf{c}\\|^2.\\]Note: A unit vector is a vector of magnitude 1.\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': "To solve this problem, we need to maximize the expression:\n\\[\n\\|\\mathbf{a} - \\mathbf{b}\\|^2 + \\|\\mathbf{a} - \\mathbf{c}\\|^2 + \\|\\mathbf{b} - \\mathbf{c}\\|^2,\n\\]\nwhere $\\mathbf{a}, \\mathbf{b}, \\mathbf{c}$ are unit vectors ($\\|\\mathbf{a}\\| = \\|\\mathbf{b}\\| = \\|\\mathbf{c}\\| = 1$). Let us break it down step by step.\n\n---\n\n### Step 1: Expand each squared norm\nFor any two vectors $\\mathbf{u}$ and $\\mathbf{v}$, the squared distance between them is given by:\n\\[\n\\|\\mathbf{u} - \\mathbf{v}\\|^2 = \\|\\mathbf{u}\\|^2 + \\|\\mathbf{v}\\|^2 - 2(\\mathbf{u} \\cdot \\mathbf{v}).\n\\]\nSince $\\|\\mathbf{a}\\| = \\|\\mathbf{b}\\| = \\|\\mathbf{c}\\| = 1$, this simplifies to:\n\\[\n\\|\\mathbf{a} - \\mathbf{b}\\|^2 = 2 - 2(\\mathbf{a} \\cdot \\mathbf{b}),\n\\]\n\\[\n\\|\\mathbf{a} - \\mathbf{c}\\|^2 = 2 - 2(\\mathbf{a} \\cdot \\mathbf{c}),\n\\]\n\\[\n\\|\\mathbf{b} - \\mathbf{c}\\|^2 = 2 - 2(\\mathbf{b} \\cdot \\mathbf{c}).\n\\]\nAdding these together gives:\n\\[\n\\|\\mathbf{a} - \\mathbf{b}\\|^2 + \\|\\mathbf{a} - \\mathbf{c}\\|^2 + \\|\\mathbf{b} - \\mathbf{c}\\|^2 = 6 - 2(\\mathbf{a} \\cdot \\mathbf{b} + \\mathbf{a} \\cdot \\mathbf{c} + \\mathbf{b} \\cdot \\mathbf{c}).\n\\]\n---\n\n### Step 2: Maximize the expression\nTo maximize the above expression, we minimize the sum $\\mathbf{a} \\cdot \\mathbf{b} + \\mathbf{a} \\cdot \\mathbf{c} + \\mathbf{b} \\cdot \\mathbf{c}$. The dot product between two unit vectors ranges from $-1$ to $1$. The minimum occurs when all three vectors are mutually orthogonal (dot products are zero):\n\\[\n\\mathbf{a} \\cdot \\mathbf{b} = \\mathbf{a} \\cdot \\mathbf{c} = \\mathbf{b} \\cdot \\mathbf{c} = -\\frac{1}{2}.\n\\]\nSubstituting this into the expression:\n\\[\n6 - 2(-\\frac{3}{2}) = 9.\n\\]\n---\n\n### Final Answer:\nThe largest possible value is:\n\\[\n\\boxed{9}\n\\]"},
{'role': 'HUMAN', 'prompt': "Problem: What is the largest five-digit integer whose digits have a product equal to the product $(7)(6)(5)(4)(3)(2)(1)$?\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': "To solve this problem, we need to determine the largest five-digit integer whose digits have a product equal to $ 7 \\times 6 \\times 5 \\times 4 \\times 3 \\times 2 \\times 1 $. Let's break it down step by step.\n\n---\n\n### Step 1: Compute the total product\nThe given product is:\n\\[\n7 \\times 6 \\times 5 \\times 4 \\times 3 \\times 2 \\times 1 = 7!.\n\\]\nCompute $ 7! $:\n\\[\n7! = 7 \\times 6 \\times 5 \\times 4 \\times 3 \\times 2 \\times 1 = 5040.\n\\]\nThus, the product of the digits of the desired five-digit number must also equal $ 5040 $.\n\n---\n\n### Step 2: Strategy to maximize the number\nTo form the largest possible five-digit integer, we want the digits to be as large as possible, arranged in descending order. Each digit must be between $ 1 $ and $ 9 $ because $ 0 $ would make the product $ 0 $, which is invalid.\n\nAdditionally, since the product of all five digits must equal $ 5040 $, we will factorize $ 5040 $ into exactly five factors (digits), prioritizing larger digits first.\n\n---\n\n### Step 3: Factorize $ 5040 $ into five digits\nStart with the prime factorization of $ 5040 $:\n\\[\n5040 = 2^4 \\times 3^2 \\times 5 \\times 7.\n\\]\nWe now group these prime factors into five integers (digits) such that their product equals $ 5040 $ and the digits are as large as possible.\n\n#### First attempt:\n- Start with the largest single-digit factor, $ 9 $ ($ 9 = 3^2 $).\n- After using $ 9 $, the remaining product is:\n\\[\n\\frac{5040}{9} = 560.\n\\]\n\n- Next, use the largest single-digit factor of $ 560 $, which is $ 8 $ ($ 8 = 2^3 $).\n- After using $ 8 $, the remaining product is:\n\\[\n\\frac{560}{8} = 70.\n\\]\n\n- Next, use the largest single-digit factor of $ 70 $, which is $ 7 $ ($ 7 = 7 $).\n- After using $ 7 $, the remaining product is:\n\\[\n\\frac{70}{7} = 10.\n\\]\n\n- Finally, split $ 10 $ into two single-digit factors: $ 5 $ and $ 2 $.\n\nThus, the five digits are $ 9, 8, 7, 5, 2 $.\n\n---\n\n### Step 4: Arrange the digits in descending order\nTo maximize the number, arrange the digits $ 9, 8, 7, 5, 2 $ in descending order:\n\\[\n98752.\n\\]\n\n---\n\n### Step 5: Verify the solution\nCheck that the product of the digits equals $ 5040 $:\n\\[\n9 \\times 8 \\times 7 \\times 5 \\times 2 = 5040.\n\\]\nThis confirms that the solution is correct.\n\n---\n\n### Final Answer:\n\\[\n\\boxed{98752}\n\\]"},
{'role': 'HUMAN', 'prompt': "Problem:\n{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
]

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
        abbr='math_qwen_enhance_2shot',
        path='opencompass/math',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg)
]
