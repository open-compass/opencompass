from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator
gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]
round = [
{'role': 'HUMAN', 'prompt': 'Question: Ivan has a bird feeder in his yard that holds two cups of birdseed. Every week, he has to refill the emptied feeder. Each cup of birdseed can feed fourteen birds, but Ivan is constantly chasing away a hungry squirrel that steals half a cup of birdseed from the feeder every week. How many birds does Ivan’s bird feeder feed weekly?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:'},
{'role': 'BOT', 'prompt': 'The squirrel steals 1/2 cup of birdseed every week, so the birds eat 2 - 1/2 = 1 1/2 cups of birdseed.\nEach cup feeds 14 birds, so Ivan’s bird feeder feeds 14 * 1 1/2 = 21 birds weekly. So the answer is $\\boxed{21}$.'},
{'role': 'HUMAN', 'prompt': 'Question: Samuel took 30 minutes to finish his homework while Sarah took 1.3 hours to finish it. How many minutes faster did Samuel finish his homework than Sarah?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:'},
{'role': 'BOT', 'prompt': 'Since there are 60 minutes in 1 hour, then 1.3 hours is equal to 1.3 x 60 = 78 minutes.\nThus, Samuel is 78 – 30 = 48 minutes faster than Sarah. So the answer is $\\boxed{48}$.'},
{'role': 'HUMAN', 'prompt': 'Question: Julia bought 3 packs of red balls, 10 packs of yellow balls, and 8 packs of green balls. There were 19 balls in each package. How many balls did Julie buy in all?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:'},
{'role': 'BOT', 'prompt': 'The total number of packages is 3 + 10 + 8 = 21.\nJulia bought 21 × 19 = 399 balls. So the answer is $\\boxed{399}$.'},
{'role': 'HUMAN', 'prompt': 'Question: Lexi wants to run a total of three and one-fourth miles. One lap on a particular outdoor track measures a quarter of a mile around. How many complete laps must she run?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:'},
{'role': 'BOT', 'prompt': 'There are 3/ 1/4 = 12 one-fourth miles in 3 miles.\nSo, Lexi will have to run 12 (from 3 miles) + 1 (from 1/4 mile) = 13 complete laps. So the answer is $\\boxed{13}$.'},
{'role': 'HUMAN', 'prompt': 'Question: Asia bought a homecoming dress on sale for $140. It was originally priced at $350. What percentage off did she get at the sale?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:'},
{'role': 'BOT', 'prompt': 'Asia saved $350 - $140 = $210 on the dress.\nThat means she saved $210 / $350 = 0.60 or 60% off on the dress. So the answer is $\\boxed{60}$.'},
{'role': 'HUMAN', 'prompt': 'Question: As a special treat, Georgia makes muffins and brings them to her students on the first day of every month.  Her muffin recipe only makes 6 muffins and she has 24 students.  How many batches of muffins does Georgia make in 9 months?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:'},
{'role': 'BOT', 'prompt': 'She has 24 students and her muffin recipe only makes 6 muffins so she needs to bake 24/6 = 4 batches of muffins\nShe brings muffins on the 1st of the month for 9 months and it takes 4 batches to feed all of her students so she bakes 9*4 = 36 batches of muffins So the answer is $\\boxed{36}$.'},
{'role': 'HUMAN', 'prompt': 'Question: Jorge bought 24 tickets for $7 each. For purchasing so many, he is given a discount of 50%. How much, in dollars, did he spend on tickets?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:'},
{'role': 'BOT', 'prompt': 'Jorge spent 24 tickets * $7 per ticket = $168 total.\nAfter applying the discount, Jorge spent $168 * 0.50 = $84. So the answer is $\\boxed{84}$.'},
{'role': 'HUMAN', 'prompt': 'Question: OpenAI runs a robotics competition that limits the weight of each robot. Each robot can be no more than twice the minimum weight and no less than 5 pounds heavier than the standard robot. The standard robot weighs 100 pounds. What is the maximum weight of a robot in the competition?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:'},
{'role': 'BOT', 'prompt': 'the minimum is 5 more than 100 so 100+5=105\nthe maximum weight of a robot is twice the minimum 105*2=210 So the answer is $\\boxed{210}$.'},
{'role': 'HUMAN', 'prompt': "Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"}
]
# que = round[-1]
# round = [round[4].copy(), round[5].copy()] * 8
# round.append(que)
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
        abbr='gsm8k_easy_v2',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
