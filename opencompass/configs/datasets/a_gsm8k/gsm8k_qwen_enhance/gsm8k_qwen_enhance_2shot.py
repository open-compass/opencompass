from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]

round = [
{'role': 'HUMAN', 'prompt': "Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "To solve this problem, let's break it down step by step:\n\n1. Understand the initial situation: \n   Initially, there are 15 trees in the grove.\n\n2. Understand the final situation: \n   After the grove workers plant additional trees, the total number of trees becomes 21.\n\n3. Determine how many trees were planted: \n   The difference between the final number of trees (21) and the initial number of trees (15) represents the number of trees planted. This can be calculated as: \n   Number of trees planted = Final number of trees - Initial number of trees\n\n4. Perform the calculation: \n   Substitute the given values into the formula: \n   Number of trees planted = 21 - 15 = 6\n\n5. Verify the result: \n   If the workers planted 6 trees, adding these to the original 15 trees gives: \n   15 + 6 = 21 \n   This matches the final number of trees stated in the problem, so the calculation is correct.\n\nThus, the number of trees planted by the grove workers today is: \n$$\n\\boxed{6}\n$$\n"},
{'role': 'HUMAN', 'prompt': "Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "To solve this problem, let's break it down step by step:\n\n1. Understand the initial situation: \n   Initially, there are 3 cars in the parking lot.\n\n2. Understand the change: \n   Two more cars arrive at the parking lot.\n\n3. Determine the total number of cars: \n   The total number of cars in the parking lot after the new arrivals is the sum of the initial number of cars and the number of cars that arrived. This can be expressed as: \n   Total number of cars = Initial number of cars + Number of cars that arrived\n\n4. Perform the calculation: \n   Substitute the given values into the formula: \n   Total number of cars = 3 + 2 = 5\n\n5. Verify the result: \n   Adding 2 cars to the initial 3 cars results in a total of 5 cars. This matches the expected outcome, so the calculation is correct.\n\nThus, the total number of cars in the parking lot is: \n$$\n\\boxed{5}\n$$\n"},
{'role': 'HUMAN', 'prompt': "Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
]

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=round)),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096, stopping_criteria=stop_words))

gsm8k_eval_cfg = dict(evaluator=dict(type=Gsm8kEvaluator),
                      pred_postprocessor=dict(type=gsm8k_postprocess),
                      dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))

gsm8k_datasets = [
    dict(
        abbr='gsm8k_qwen_enhance_2shot',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
