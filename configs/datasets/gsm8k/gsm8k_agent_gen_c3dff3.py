from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AgentInferencer
from opencompass.datasets import (
    GSM8KDataset,
    gsm8k_postprocess,
    gsm8k_dataset_postprocess,
    Gsm8kAgentEvaluator,
)

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                # # ################################### NEW SHOT ###################################
                dict(role='HUMAN', prompt='Mark\'s basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What\'s the total number of points scored by both teams added together?'),
                dict(role='BOT', prompt='Tool:PythonInterpreter\nTool Input:```python\ndef solution():\n    mark_pointers_2 = 25 * 2\n    mark_pointers_3 = 8 * 3\n    mark_free_throws = 10 * 1\n    mark_points_scored = mark_pointers_2 + mark_pointers_3 + mark_free_throws\n    opponents_pointers_2 = mark_pointers_2 * 2\n    opponents_pointers_3 = mark_pointers_3 / 2\n    opponents_free_throws = mark_free_throws / 2\n    opponents_points_scored = opponents_pointers_2 + opponents_pointers_3 + opponents_free_throws\n    total_points_scored = mark_points_scored + opponents_points_scored\n    result = total_points_scored\n    return result\n```'),
                dict(role='SYSTEM', prompt='Response:201'),
                dict(role='BOT', prompt='Thought: According to the response, I got the answer\nFinalAnswer: 201'),

                dict(role='HUMAN', prompt='Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?'),
                dict(role='BOT', prompt='Tool:PythonInterpreter\nTool Input:```python\ndef solution():\n    marbles = 60\n    num_increased_marbles = marbles * 2 / 5\n    num_total_marbles = marbles + num_increased_marbles\n    frisbees = marbles / 2\n    num_increased_frisbees = frisbees * 2 / 5\n    num_total_frisbees = frisbees + num_increased_frisbees\n    deck_cards = frisbees - 20\n    num_increased_deck_cards = deck_cards * 2 / 5\n    num_total_deck_cards = deck_cards + num_increased_deck_cards\n    num_total = num_total_marbles + num_total_frisbees + num_total_deck_cards\n    result = num_total\n    return result\n```'),
                dict(role='SYSTEM', prompt='Response:140'),
                dict(role='BOT', prompt='Thought: According to the response, I got the answer\nFinalAnswer: 140'),

                dict(role='HUMAN', prompt='A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?'),
                dict(role='BOT', prompt="""Tool:PythonInterpreter\nTool Input:```python\ndef solution():\n    num_fruits_per_first_three_basket = 9 + 15 + 14\n    num_fruits_first_three_basket = num_fruits_per_first_three_basket * 3\n    num_apple_fourth_basket = 9 - 2\n    num_orange_fourth_basket = 15 - 2\n    num_banana_fourth_basket = 14 - 2\n    num_fruits_fourth_basket = num_apple_fourth_basket + num_orange_fourth_basket + num_banana_fourth_basket\n    num_fruits_total = num_fruits_first_three_basket + num_fruits_fourth_basket\n    result = num_fruits_total\n    return result\n```"""),
                dict(role='SYSTEM', prompt='Response:146'),
                dict(role='BOT', prompt='Thought: According to the response, I got the answer\nFinalAnswer: 146'),

                dict(role='HUMAN', prompt='{question}'),
            ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AgentInferencer),
)

gsm8k_eval_cfg = dict(
    evaluator=dict(type=Gsm8kAgentEvaluator),
    pred_postprocessor=dict(type=gsm8k_postprocess),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
)

gsm8k_datasets = [
    dict(
        abbr='gsm8k-agent',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
