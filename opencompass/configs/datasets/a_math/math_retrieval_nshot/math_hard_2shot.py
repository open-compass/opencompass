from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess_v2

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","<|end_of_text|>","\nught","\nughtuser","\nQuestion","\nnoteq","\nosterone","nosteroneuser"]
round = [
{'role': 'HUMAN', 'prompt': "Problem: If $\\mathbf{a},$ $\\mathbf{b},$ and $\\mathbf{c}$ are unit vectors, then find the largest possible value of\n\\[\\|\\mathbf{a} - \\mathbf{b}\\|^2 + \\|\\mathbf{a} - \\mathbf{c}\\|^2 + \\|\\mathbf{b} - \\mathbf{c}\\|^2.\\]Note: A unit vector is a vector of magnitude 1.\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': 'We can write\n\\begin{align*}\n\\|\\mathbf{a} - \\mathbf{b}\\|^2 &= (\\mathbf{a} - \\mathbf{b}) \\cdot (\\mathbf{a} - \\mathbf{b}) \\\\\n&= \\mathbf{a} \\cdot \\mathbf{a} - 2 \\mathbf{a} \\cdot \\mathbf{b} + \\mathbf{b} \\cdot \\mathbf{b} \\\\\n&= \\|\\mathbf{a}\\|^2 - 2 \\mathbf{a} \\cdot \\mathbf{b} + \\|\\mathbf{b}\\|^2 \\\\\n&= 2 - 2 \\mathbf{a} \\cdot \\mathbf{b}.\n\\end{align*}Similarly, $\\|\\mathbf{a} - \\mathbf{c}\\|^2 = 2 - 2 \\mathbf{a} \\cdot \\mathbf{c}$ and $\\|\\mathbf{b} - \\mathbf{c}\\|^2 = 2 - 2 \\mathbf{b} \\cdot \\mathbf{c},$ so\n\\[\\|\\mathbf{a} - \\mathbf{b}\\|^2 + \\|\\mathbf{a} - \\mathbf{c}\\|^2 + \\|\\mathbf{b} - \\mathbf{c}\\|^2 = 6 - 2 (\\mathbf{a} \\cdot \\mathbf{b} + \\mathbf{a} \\cdot \\mathbf{c} + \\mathbf{b} \\cdot \\mathbf{c}).\\]Now,\n\\[\\|\\mathbf{a} + \\mathbf{b} + \\mathbf{c}\\|^2 \\ge 0.\\]We can expand this as\n\\[\\|\\mathbf{a}\\|^2 + \\|\\mathbf{b}\\|^2 + \\|\\mathbf{c}\\|^2 + 2 \\mathbf{a} \\cdot \\mathbf{b} + 2 \\mathbf{a} \\cdot \\mathbf{c} + 2 \\mathbf{b} \\cdot \\mathbf{c} \\ge 0.\\]Then $2 (\\mathbf{a} \\cdot \\mathbf{b} + \\mathbf{a} \\cdot \\mathbf{c} + \\mathbf{b} \\cdot \\mathbf{c}) \\ge -3,$ so\n\\[\\|\\mathbf{a} - \\mathbf{b}\\|^2 + \\|\\mathbf{a} - \\mathbf{c}\\|^2 + \\|\\mathbf{b} - \\mathbf{c}\\|^2 = 6 - 2 (\\mathbf{a} \\cdot \\mathbf{b} + \\mathbf{a} \\cdot \\mathbf{c} + \\mathbf{b} \\cdot \\mathbf{c}) \\le 9.\\]Equality occurs when $\\mathbf{a},$ $\\mathbf{b},$ and $\\mathbf{c}$ are equally spaced on a circle with radius 1 (where $\\|\\mathbf{a} - \\mathbf{b}\\| = \\|\\mathbf{a} - \\mathbf{c}\\| = \\|\\mathbf{b} - \\mathbf{c}\\| = \\sqrt{3}$), so the largest possible value is $\\boxed{9}.$\n'},
{'role': 'HUMAN', 'prompt': "Problem: What is the largest five-digit integer whose digits have a product equal to the product $(7)(6)(5)(4)(3)(2)(1)$?\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': "Since we want a large 5-digit integer, we want the digits to the left to be as large as possible. We prime factorize the product to get $7 \\cdot 5 \\cdot 3^2 \\cdot 2^4$. The largest single-digit number is $9$, which can be found by $3^2$. This leaves us $7 \\cdot 5 \\cdot 2^4$. We can get the next largest number, $8$, by using $2^3$. This leaves $7 \\cdot 5\\cdot 2$. We can't multiply any of those numbers together to get a single-digit number, so we have that the remaining three digits are $7,5$, and $2$. Ordering the digits from largest to smallest, we get $\\boxed{98752}$.\n"},
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
        abbr='math_hard_2shot',
        path='opencompass/math',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg)
]
