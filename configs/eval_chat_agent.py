from mmengine.config import read_base
from opencompass.models.openai_api import OpenAI
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.openicl import AgentInferencer

with read_base():
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets as datasets

from opencompass.models.lagent import LagentAgent
from lagent import PythonInterpreter, ReAct
from lagent.agents.react import ReActProtocol

FORCE_STOP_PROMPT_EN = """You should directly give results based on history information."""

FEWSHOT_INSTRUCTION = """\
You are a assistant who can utilize external tools.
{tool_description}
To use a tool, please use the following format:
```
{thought} Think what you need to solve, do you need to use tools?
{action} the tool name, should be one of [{action_names}]
{action_input} the input to the action
```
I will give you response after utilizing tools should using the following format:
```
{response} the results after call the tool.
``
If you already know the answer, or you do not need to use tools,
please using the following format to reply:
```
{thought} the thought process to get the final answer
{finish} final answer
```

Examples:

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

Begin!
"""

PYTHON_INTERPRETER_DESCRIPTION = '''\
It can run a Python code. The code must be a valid code that contains only python method, and the method' name must be 'solution' and returns a dict, which key is variable name. The libraries I recommend are sympy and scipy. the format is:
```python
# import packages
import xxx
def solution():
    # initialize some variables
    variable_names_with_real_meaning = xxx
    # middle steps
    mid_variable = func(mid_variable)
    # final answer
    final_answer =  func(mid_variable)
    return final_answer
```'''

models = [
    dict(
        abbr='gpt-3.5-react',
        type=LagentAgent,
        agent_type=ReAct,
        max_turn=3,
        llm=dict(
            type=OpenAI,
            path='gpt-3.5-turbo',
            key='ENV',
            query_per_second=1,
            max_seq_len=4096,
        ),
        actions=[
            dict(type=PythonInterpreter,
                 description=PYTHON_INTERPRETER_DESCRIPTION),
        ],
        protocol=dict(
            type=ReActProtocol,
            call_protocol=FEWSHOT_INSTRUCTION,
            force_stop=FORCE_STOP_PROMPT_EN,
            finish=dict(role='FINISH', begin='Final Answer:', end='\n'),
        ),
        batch_size=1,
    ),
]

for dataset in datasets:
    # Use AgentInferencer instead of GenInferencer
    dataset['infer_cfg']['inferencer'] = dict(type=AgentInferencer)
    # Use the question as agent input directly.
    dataset['infer_cfg']['prompt_template']['template'] = "{question}"

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLInferTask)),
)
