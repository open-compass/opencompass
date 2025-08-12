from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator


gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt="Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nPlease provide your answer directly without additional reasoning steps. Format your answer as 'The answer is (insert answer here)'\nAnswer:"),
                dict(role='BOT', prompt='The answer is 4\n'),
                dict(role='HUMAN', prompt="Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?\nPlease provide your answer directly without additional reasoning steps. Format your answer as 'The answer is (insert answer here)'\nAnswer:"),
                dict(role='BOT', prompt="The answer is 201\n"),
                dict(role='HUMAN', prompt="Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?\nPlease provide your answer directly without additional reasoning steps. Format your answer as 'The answer is (insert answer here)'\nAnswer:"),
                dict(role='BOT', prompt="The answer is 140\n"),
                dict(role='HUMAN', prompt="Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?\nPlease provide your answer directly without additional reasoning steps. Format your answer as 'The answer is (insert answer here)'\nAnswer:"),
                dict(role='BOT', prompt='The answer is 146\n'),
                dict(role='HUMAN', prompt="Question: {question}\nPlease provide your answer directly without additional reasoning steps. Format your answer as 'The answer is (insert answer here)'\nAnswer:"),
            ],
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

gsm8k_eval_cfg = dict(evaluator=dict(type=Gsm8kEvaluator),
                      pred_postprocessor=dict(type=gsm8k_postprocess),
                      dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))

gsm8k_datasets = [
    dict(
        abbr='gsm8k',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
