from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]
# Qwen2.5-7B
round = [
{'role': 'HUMAN', 'prompt': "Question: Pulsar, the shuffle-dancing bear, Polly, the pink prancing poodle, and Petra, the proud portly pachyderm, are entertainers at the Big Top Circus. In one show, Pulsar stands on his two back legs for a total of 10 minutes. Then, Polly stands on her back legs for three times as long as Pulsar. And then, finally, Petra stands on his back legs for one-sixth as long as Polly. What is the combined length of time, in minutes, that the three entertainers stand on their back legs?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "Pulsar stands on his back legs for 10 minutes.\nPolly stands for 3 times that, so 10 × 3 = 30 minutes.\nPetra stands for one-sixth as long as Polly, so 30 ÷ 6 = 5 minutes.\nIn total, they stand for 10 + 30 + 5 = 45 minutes.\nSo the answer is $\\boxed{45}$.\n"},
{'role': 'HUMAN', 'prompt': "Question: The cost of a slice of cake is three-fourths of the cost of a cup of milk tea. If the milk tea costs $2.40, how much do 2 slices of cake and 1 cup of milk tea cost?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "The cost of a slice of cake is 3/4 × $2.40 = $1.80.\nTwo slices of cake cost 2 × $1.80 = $3.60.\nAdding the cost of 1 cup of milk tea: $3.60 + $2.40 = $6.00.\nSo the answer is $\\boxed{6}$.\n"},
{'role': 'HUMAN', 'prompt': "Question: The bowling alley has 30 red bowling balls. There are 6 more green bowling balls than red. How many bowling balls are there in total?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "There are 30 red bowling balls.\nThere are 6 more green balls than red, so 30 + 6 = 36 green balls.\nTotal number of balls is 30 + 36 = 66.\nSo the answer is $\\boxed{66}$.\n"},
{'role': 'HUMAN', 'prompt': "Question: The chances of making the junior high basketball team start at 10% if you're 66 inches and increase 10% for every additional inch of height. Devin starts out as 65 inches tall, then grows 3 inches. What are his chances of making the basketball team?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "Devin's original height is 65 inches. He grows 3 inches, so his new height is 65 + 3 = 68 inches.\nCompared to 66 inches, Devin is now 2 inches taller.\nEach inch increases the chance by 10%, so 2 × 10% = 20% increase.\nBase chance is 10%, so final chance is 10% + 20% = 30%.\nSo the answer is $\\boxed{30\\%}$.\n"},
{'role': 'HUMAN', 'prompt': "Question: Allyn is a golfer. At the starting tee, he hit the golf ball and it traveled 180 yards straight toward the hole. On his second turn, he hit the ball again straight toward the hole and it traveled half as far as it did on his first turn, but the ball landed 20 yards beyond the hole. On his third swing, he hit the ball onto the green and it rolled into the hole. What is the distance, in yards, from the starting tee to the hole?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "On the first turn, Allyn hit the ball 180 yards.\nOn the second turn, he hit it half as far: 180 ÷ 2 = 90 yards.\nSo after two turns, the total distance traveled is 180 + 90 = 270 yards.\nSince the ball went 20 yards past the hole, the distance from the tee to the hole is 270 − 20 = 250 yards.\nSo the answer is $\\boxed{250}$.\n"},
{'role': 'HUMAN', 'prompt': "Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
]

# llama3.1-8B
# round = [
# {'role': 'HUMAN', 'prompt': "Question: Jessie won 3 times as many athletic awards as his buddy Scott, who won 4 awards. The best athlete at the rival high school won twice as many awards as Jessie. How many awards did the rival win?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
# {'role': 'BOT', 'prompt': "Scott won 4 awards.\nJessie won 3 times that, so 3 × 4 = 12 awards.\nThe rival athlete won twice as many as Jessie, so 2 × 12 = 24 awards.\nSo the answer is $\\boxed{24}$.\n"},
# {'role': 'HUMAN', 'prompt': "Question: Inez has $150. She spends one-half on hockey skates and a certain amount on hockey pads. If Inez has $25 remaining, how much did the hockey pads cost, together, in dollars?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
# {'role': 'BOT', 'prompt': "She spent 150 ÷ 2 = 75 dollars on hockey skates.\nLet x be the amount spent on hockey pads.\nThen 150 - 75 - x = 25 ⇒ 75 - x = 25 ⇒ x = 50.\nSo the answer is $\\boxed{50}$.\n"},
# {'role': 'HUMAN', 'prompt': "Question: On a road map, 1/4 inch represents 8 miles of actual road distance. The towns of Pence and Hillcrest are represented by points 3 3/8 inches apart on the map. What is the actual distance, in miles, between the towns?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
# {'role': 'BOT', 'prompt': "1/4 inch represents 8 miles, so 1 inch = 4 × 8 = 32 miles.\n3 inches = 3 × 32 = 96 miles.\n3/8 inch = (3/8) ÷ (1/4) = 1.5 segments of 8 miles = 3 × 8 ÷ 2 = 12 miles.\nTotal distance = 96 + 12 = 108 miles.\nSo the answer is $\\boxed{108}$.\n"},
# {'role': 'HUMAN', 'prompt': "Question: Cristina, John, Clarissa and Sarah want to give their mother a photo album for her birthday. Cristina brings 7 photos, John brings 10 photos and Sarah brings 9 photos. If the photo album has 40 slots available, how many photos does Clarissa need to bring in order to complete the photo album?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
# {'role': 'BOT', 'prompt': "Cristina, John, and Sarah bring a total of 7 + 10 + 9 = 26 photos.\nThe album has 40 slots, so Clarissa must bring 40 - 26 = 14 photos.\nSo the answer is $\\boxed{14}$.\n"},
# {'role': 'HUMAN', 'prompt': "Question: James turned 23 the same time John turned 35. Tim is 5 years less than twice John's age. If Tim is 79 how old is James?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
# {'role': 'BOT', 'prompt': "The age gap between John and James is 35 - 23 = 12 years.\nTim is 5 years less than twice John's age: 2x - 5 = 79 ⇒ 2x = 84 ⇒ x = 42.\nSo John is 42, and James is 42 - 12 = 30.\nSo the answer is $\\boxed{30}$.\n"},
# {'role': 'HUMAN', 'prompt': "Question: Jenny is planning her catering budget for her wedding. She's going to have 80 guests. 3 times as many guests want steak as chicken. If each steak entree costs $25 and each chicken entree costs $18, how much is the total catering budget?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
# {'role': 'BOT', 'prompt': "Let c be the number of chicken meals. Then steak meals = 3c.\nTotal meals: c + 3c = 4c = 80 ⇒ c = 20.\nChicken cost: 20 × 18 = 360.\nSteak meals: 3 × 20 = 60 ⇒ cost = 60 × 25 = 1500.\nTotal budget = 1500 + 360 = 1860.\nSo the answer is $\\boxed{1860}$.\n"},
# {'role': 'HUMAN', 'prompt': "Question: Janet has a business selling custom collars for dogs and cats. If it takes 18 inches of nylon to make a dog collar and 10 inches to make a cat collar, how much nylon does she need to make 9 dog collars and 3 cat collars?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
# {'role': 'BOT', 'prompt': "Nylon for dog collars: 9 × 18 = 162 inches.\nNylon for cat collars: 3 × 10 = 30 inches.\nTotal nylon needed = 162 + 30 = 192 inches.\nSo the answer is $\\boxed{192}$.\n"},
# {'role': 'HUMAN', 'prompt': "Question: Kenny wants to make sure he does more jumping jacks every week than he did the week before. He recorded that last week he did 324 total jumping jacks. On Saturday of this week, he looks at his records and sees that on Sunday he did 34. On Monday he did 20. On Tuesday he skipped a day. On Wednesday he did 123. On Thursday he did 64. On Friday he did 23. How many does he have to do on Saturday to make sure he beats last week's number?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
# {'role': 'BOT', 'prompt': "Total done so far: 34 + 20 + 123 + 64 + 23 = 264.\nLast week's total: 324 ⇒ to beat it he needs >324.\n324 - 264 = 60, so he needs more than 60 on Saturday.\nMinimum required: 60 + 1 = 61.\nSo the answer is $\\boxed{61}$.\n"},
# {'role': 'HUMAN', 'prompt': "Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
# ]


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
        abbr='gsm8k_explora',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
