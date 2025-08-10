from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AgentInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kAgentEvaluator
# This config is for code interpreter
gsm8k_example = """
Example:

<HUMAN>A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?
<ASSISTANT>{thought} We need to calculate the total number of fruits. The total number of fruits in the first three baskets is given, while for the fourth basket, we need to subtract 2 from each fruit category. We can solve this problem using simple arithmetic.
{action} PythonInterpreter
{action_input}
```python
def solution():
    # Fruits in the first three baskets
    apples_first_three = 9
    oranges_first_three = 15
    bananas_first_three = 14

    # Fruits in the fourth basket
    apples_fourth = apples_first_three - 2
    oranges_fourth = oranges_first_three - 2
    bananas_fourth = bananas_first_three - 2

    # Total fruits
    total_fruits = ((apples_first_three + oranges_first_three + bananas_first_three) * 3 +
                    apples_fourth + oranges_fourth + bananas_fourth)

    return {{"total_fruits": total_fruits}}
```
<SYSTEM>{response}{{'total_fruits': 146}}
<ASSISTANT> {thought} By adding the given numbers of apples, oranges, and bananas in the first three baskets, then subtracting 2 from each category for the fourth basket, we have found the total number of fruits.
{finish} 146

<HUMAN>Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?
<ASSISTANT>{thought} This is a problem that requires solving equations. We know the relationship between the number of marbles, frisbees, and deck cards. Bella has twice as many marbles as frisbees, and 20 more frisbees than deck cards. Finally, we are told Bella buys 2/5 times more of each item. This purchasing will increase the number of each type of item.
{action} PythonInterpreter
{action_input}
```python
def solution():
    # Given number of marbles
    marbles_now = 60

    # Calculate number of frisbees and deck cards now
    frisbees_now = marbles_now / 2
    cards_now = frisbees_now - 20

    # Calculate number of each item after buying more
    marbles_then = marbles_now + (2/5) * marbles_now
    frisbees_then = frisbees_now + (2/5) * frisbees_now
    cards_then = cards_now + (2/5)*cards_now

    # Total number of items then
    total_items = marbles_then + frisbees_then + cards_then

    return {{"total_items": total_items}}
```
<SYSTEM>{response}{{'total_items': 140.0}}
<ASSISTANT>{thought} By establishing the relationships between the numbers of marbles, frisbees, and deck cards that Bella currently has, we can calculate how many of each item she will have after buying 2/5 more of each. Adding these quantities together gives us the total number of items.
{finish} 140
"""

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template='{question}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AgentInferencer, example=gsm8k_example))

gsm8k_eval_cfg = dict(
    evaluator=dict(type=Gsm8kAgentEvaluator),
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
