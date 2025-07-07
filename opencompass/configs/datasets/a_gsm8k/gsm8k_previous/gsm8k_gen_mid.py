from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator
gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]
round = [
{'role': 'HUMAN', 'prompt': "Question: In 5 years, Heath will be 3 times as old as Jude.  If Heath is 16 years old today, how old is Jude today?\nLet's think step by step\nAnswer:"},
{'role': 'BOT', 'prompt': 'In 5 years, Heath will be 16 + 5 = 21 years old.\nIn 5 years, Jude will be 21 / 3 = 7 years old.\nThe difference in their ages is 21 - 7 = 14 years.\nToday, Jude will be 16 - 14 = 2 years old. So the answer is 2.\n'},
{'role': 'HUMAN', 'prompt': "Question: Mancino is tending 3 gardens that each measure 16 feet by 5 feet. His sister, Marquita, is tilling the soil for two gardens that each measure 8 feet by 4 feet. How many square feet combined are in all their gardens?\nLet's think step by step\nAnswer:"},
{'role': 'BOT', 'prompt': '3 * (16 * 5) = 240 square feet\n2 * (8 * 4) = 64 square feet\n240 + 64 = 304 square feet\nThe gardens have a total of 304 square feet. So the answer is 304.\n'},
{'role': 'HUMAN', 'prompt': "Question: Bridge and Sarah have $3 between them. If Bridget has 50 cents more than Sarah, how many cents does Sarah have?\nLet's think step by step\nAnswer:"},
{'role': 'BOT', 'prompt': 'Let X be the amount of money Sarah has, so Bridget has X + 50 cents\nWe know that Sarah and Bridget combined have $3, or 300 cents, so X + X + 50 = 300\nWe therefore know that 2X = 250\nSo Sarah has 250 / 2 = 125 cents So the answer is 125.\n'},
{'role': 'HUMAN', 'prompt': "Question: Ben has four boxes with ten basketball cards in each box. His mother gave him five boxes with eight baseball cards.  If he gives 58 cards to his classmates, how many cards does he has left?\nLet's think step by step\nAnswer:"},
{'role': 'BOT', 'prompt': 'Ben has 4 x 10 = 40 basketball cards.\nHis mother gave him 5 x 8 = 40 baseball cards.\nThus, Ben has 40 + 40 = 80 cards in all.\nTherefore, Ben has 80 - 58 = 22 cards left. So the answer is 22.\n'},
{'role': 'HUMAN', 'prompt': "Question: Lulu has $65 in her piggy bank. She spent $5 on ice cream. She then spent half of the remainder of the money on a t-shirt. Afterwards, she went to the bank and deposited a fifth of her remaining money. How much cash was Lulu left with?\nLet's think step by step\nAnswer:"},
{'role': 'BOT', 'prompt': 'Lulu had 65-5 = 60 dollars left after eating ice cream.\nLulu had 60/2 = 30 dollars left.\nLulu deposited 30/5 = 6 dollars\nLulu was left with 30-6 = 24 in cash. So the answer is 24.\n'},
{'role': 'HUMAN', 'prompt': "Question: Kim orders a $10 meal and a drink for 2.5.  She gives a 20% tip.  She pays with a $20 bill.  How much does she get in change?\nLet's think step by step\nAnswer:"},
{'role': 'BOT', 'prompt': 'The dinner cost 10+2.5=$12.5.\nThe tip comes to 12.5*.2=$2.5.\nSo the total cost is 12.5+2.5=$15.\nThat means she gets 20-15=$5 in change. So the answer is 5.\n'},
{'role': 'HUMAN', 'prompt': "Question: A gallon of whole milk that normally costs $3 is now sold at $2. A box of cereal was sold at a discount of $1. How much will you save via discounts if you buy 3 gallons of whole milk and 5 boxes of cereal?\nLet's think step by step\nAnswer:"},
{'role': 'BOT', 'prompt': 'The discount for each gallon of whole milk is $3 - $2 = $1.\nSo the discount for 3 gallons of whole milk is $1/gallon x 3 gallons = $3.\nThe total discount for 5 boxes of cereal is $1/box x 5 boxes = $5.\nYou will save $3 + $5 = $8 for 3 gallons of whole milk and 5 boxes of cereal. So the answer is 8.\n'},
{'role': 'HUMAN', 'prompt': "Question: There are 32 students in a statistics course.  25 percent of the class received an A.  1/4 of the remaining students got either a B or C, and the rest of the students failed.  How many students failed?\nLet's think step by step\nAnswer:"},
{'role': 'BOT', 'prompt': 'A grade:32(.25)=8\n32-8=24 students remaining\nB/C grade:24/4=6\n24-6=18 students fail So the answer is 18.\n'},
{'role': 'HUMAN', 'prompt': "Question: {question}\nLet's think step by step\nAnswer:"},
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
        abbr='gsm8k_mid',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
