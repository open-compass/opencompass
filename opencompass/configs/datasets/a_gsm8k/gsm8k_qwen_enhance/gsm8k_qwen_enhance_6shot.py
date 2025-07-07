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
{'role': 'HUMAN', 'prompt': "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "To solve this problem, let's break it down step by step:\n\n1. Understand the initial situation: \n   Leah initially has 32 chocolates, and her sister has 42 chocolates.\n\n2. Calculate the total number of chocolates initially: \n   The total number of chocolates they both have together is: \n   Total chocolates initially = Leah's chocolates + Sister's chocolates \n   Substituting the values: \n   Total chocolates initially = 32 + 42 = 74\n\n3. Understand the change: \n   They ate 35 chocolates.\n\n4. Determine the remaining chocolates: \n   The number of chocolates left is the total number of chocolates initially minus the number of chocolates eaten. This can be expressed as: \n   Chocolates left = Total chocolates initially - Chocolates eaten \n   Substituting the values: \n   Chocolates left = 74 - 35 = 39\n\n5. Verify the result: \n   Starting with 74 chocolates and eating 35 leaves 39 chocolates. This matches the expected outcome, so the calculation is correct.\n\nThus, the total number of chocolates they have left is: \n$$\n\\boxed{39}\n$$\n"},
{'role': 'HUMAN', 'prompt': "Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "To solve this problem, let's break it down step by step:\n\n1. Understand the initial situation: \n   Jason initially has 20 lollipops.\n\n2. Understand the final situation: \n   After giving some lollipops to Denny, Jason has 12 lollipops left.\n\n3. Determine how many lollipops Jason gave to Denny: \n   The difference between the initial number of lollipops (20) and the final number of lollipops (12) represents the number of lollipops Jason gave away. This can be calculated as: \n   Lollipops given to Denny = Initial number of lollipops - Final number of lollipops\n\n4. Perform the calculation: \n   Substitute the given values into the formula: \n   Lollipops given to Denny = 20 - 12 = 8\n\n5. Verify the result: \n   If Jason gave Denny 8 lollipops, subtracting these from his initial 20 lollipops leaves: \n   20 - 8 = 12 \n   This matches the final number of lollipops stated in the problem, so the calculation is correct.\n\nThus, the number of lollipops Jason gave to Denny is: \n$$\n\\boxed{8}\n$$\n"},
{'role': 'HUMAN', 'prompt': "Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "To solve this problem, let's break it down step by step:\n\n1. Understand the initial situation: \n   Shawn initially has 5 toys.\n\n2. Understand the change: \n   Shawn received 2 toys from his mom and 2 toys from his dad, for a total of: \n   Toys received = 2 + 2 = 4\n\n3. Determine the total number of toys Shawn has now: \n   The total number of toys Shawn has now is the sum of his initial toys and the toys he received. This can be expressed as: \n   Total toys now = Initial toys + Toys received \n   Substituting the values: \n   Total toys now = 5 + 4 = 9\n\n4. Verify the result: \n   Adding 4 toys to the initial 5 toys results in a total of 9 toys. This matches the expected outcome, so the calculation is correct.\n\nThus, the total number of toys Shawn has now is: \n$$\n\\boxed{9}\n$$\n"},
{'role': 'HUMAN', 'prompt': "Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "To solve this problem, let's break it down step by step:\n\n1. Understand the initial situation: \n   Initially, there are 9 computers in the server room.\n\n2. Understand the change: \n   From Monday to Thursday (4 days), 5 computers are installed each day. The total number of computers installed over these 4 days is: \n   Total computers installed = 5 x 4 = 20\n\n3. Determine the total number of computers in the server room now: \n   The total number of computers in the server room now is the sum of the initial computers and the computers installed. This can be expressed as: \n   Total computers now = Initial computers + Computers installed \n   Substituting the values: \n   Total computers now = 9 + 20 = 29\n\n4. Verify the result: \n   Adding 20 computers to the initial 9 computers results in a total of 29 computers. This matches the expected outcome, so the calculation is correct.\n\nThus, the total number of computers in the server room now is: \n$$\n\\boxed{29}\n$$\n"},
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
        abbr='gsm8k_qwen_enhance_6shot',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
