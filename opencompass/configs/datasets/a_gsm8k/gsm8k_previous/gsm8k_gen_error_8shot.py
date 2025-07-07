from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]
round = [
{'role': 'HUMAN', 'prompt': "Question: Jeanette is practicing her juggling. Each week she can juggle 2 more objects than the week before. If she starts out juggling 3 objects and practices for 5 weeks, how many objects can she juggle?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': 'Jeanette starts with 3 objects. Each week, she adds 2 objects. After 5 weeks, she gains 2 * 5 = 10 additional objects. Adding her starting amount: 3 + 10 = 13. So the answer is $\\boxed{13}$.\n'},
{'role': 'HUMAN', 'prompt': "Question: 40% of the mosquitos in Jack's area are infected with malaria. 20% of the mosquitos are infected with Zika virus. Without a vaccine, the chances of getting infected with either virus after getting bitten by an infected mosquito are 50%. Jack is taking an experimental malaria vaccine that reduces the chances of getting infected after getting bitten by 50%. If Jack gets bitten by a random mosquito, what is the percentage chance he catches either Zika virus or malaria?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': 'The total percentage of infected mosquitos is 40% + 20% = 60%. Without the vaccine, the infection chance is 60% * 50% = 30%. The malaria vaccine reduces this risk by 50%, so the final chance is 30% * 50% = 15%. So the answer is $\\boxed{15}$%.\n'},
{'role': 'HUMAN', 'prompt': "Question: Joseph invested $1000 into a hedge fund. The fund promised a yearly interest rate of 10%. If he deposited an additional $100 every month into the account to add to his initial investment of $1000, how much money will he have in the fund after two years?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': 'After the first year, Joseph’s total investment is $1000 + ($100 * 12) = $2200. With 10% interest, it grows to $2200 * 1.10 = $2420. In the second year, he adds another $1200, making $3620. With 10% interest, it becomes $3620 * 1.10 = $3982. So the answer is $\\boxed{3982}$.\n'},
{'role': 'HUMAN', 'prompt': "Question: Dexter and Jay are using a gallon of white paint to paint the walls. Dexter used 3/8 of the gallon of paint while Jay used 5/8 of a gallon of paint. If a gallon is equal to 4 liters, how many liters of paint were left from Dexter and Jay combined?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': 'Dexter used (3/8) * 4 = 1.5 liters. Jay used (5/8) * 4 = 2.5 liters. Together, they used 1.5 + 2.5 = 4 liters. Since they started with 2 gallons (8 liters), 8 - 4 = 4 liters remain. So the answer is $\\boxed{4}$.\n'},
{'role': 'HUMAN', 'prompt': "Question: Each week, Paul has 2 hours of homework on weeknights and 5 hours for the entire weekend. This week Paul has practice 2 nights out of the week and can't do any homework those nights. How many hours of homework does he have to average for the other nights to get his week's homework done?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': 'Total weekday homework: 2 * 5 = 10 hours. Total homework: 10 + 5 = 15 hours. With 2 nights unavailable, he has 5 days left. He must average 15 / 5 = 3 hours per night. So the answer is $\\boxed{3}$.\n'},
{'role': 'HUMAN', 'prompt': "Question: Chris wants to hold his breath underwater for 90 straight seconds so he starts training each day. On the first day, he holds it for 10 seconds. On the second day, he holds it for 20 seconds. On the third day, he holds it for 30 seconds. After realizing that he can hold it for ten extra seconds each day, he realizes he can calculate how many days it is until he reaches 90 if he keeps up the same pace. So how many days until he can hold his breath for 90 seconds?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': 'Chris already holds his breath for 30 seconds on day 3. To reach 90 seconds, he needs an additional 90 - 30 = 60 seconds. At a rate of 10 seconds improvement per day, this requires 60 / 10 = 6 days. So the answer is $\\boxed{6}$.\n'},
{'role': 'HUMAN', 'prompt': "Question: The middle school sold 6 more than two times the number of fair tickets as it did tickets to the baseball game. If 25 fair tickets were sold, how many baseball game tickets did the school sell?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': 'Let B be the number of baseball tickets. According to the problem, fair tickets = 2B + 6. Given 25 fair tickets: 2B + 6 = 25 → 2B = 19 → B = 9.5. Since partial tickets are impossible, the problem likely contains an error. Assuming integer solutions, the closest valid answer is $\\boxed{9}$.\n'},
{'role': 'HUMAN', 'prompt': "Question: A hay farmer harvested 560 bales of hay from 5 acres of grass per month last year. This year, he planted an additional 7 acres of grass. If the farmer also owns 9 horses and each horse consumes 3 bales of hay a day, how many bales of hay would the farmer have left by the end of December if he starts feeding them this year's hay beginning the first day of September?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': 'Monthly production per acre: 560 / 5 = 112 bales. Total acres this year: 5 + 7 = 12. Monthly production: 12 * 112 = 1344 bales. Annual production: 1344 * 12 = 16128 bales. September to December has 122 days. Daily horse consumption: 9 * 3 = 27 bales. Total consumed: 27 * 122 = 3294 bales. Remaining bales: 16128 - 3294 = 12834. So the answer is $\\boxed{12834}$.\n'},
{'role': 'HUMAN', 'prompt': "Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"}
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
        abbr='gsm8k_error_8shot',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
