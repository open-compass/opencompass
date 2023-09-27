from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.openicl import AgentInferencer
from opencompass.models import OpenAI, HuggingFaceCausalLM

with read_base():
    from .datasets.ds1000.ds1000_gen_11111 import ds1000_datasets as datasets

from opencompass.models.lagent import LagentAgent
from lagent.llms import GPTAPI
from lagent.agents.react import ReAct, ReActProtocol
# from lagent.actions import PythonInterpreter
from opencompass.lagent.actions.python_interpreter import PythonInterpreter

FORCE_STOP_PROMPT_EN = """You should directly give results based on history information."""

FEWSHOT_INSTRUCTION = """\
You are a assistant who can utilize external tools.
{tool_description}
To use a tool, please response with the following format:
```
{thought} Think what you need to solve, do you need to use tools?
{action} The tool name, should be one of [{action_names}].
{action_input} The input to the action tool.
```
The action will give you response after utilizing tools using the following format, you do not generate this part:
```
{response} The results after call the tool.
```
If you already know the answer, or you do not need to use tools, please using the following format to reply:
```
{thought} The thought process to get the final answer
{finish} Final answer
```

In the following task, you should generate code with one assertion to testify the correctness of your code.

Examples:

<HUMAN>Problem:
How do I get the dimensions of an array? For instance, this is (2, 2):
a = np.array([[1,2],[3,4]])

<ASSISTANT>{thought} In Python, Numpy provides a method called `shape` which helps to get the dimensions of an array.
{action} PythonInterpreter
{action_input}
```python
import numpy as np

def solution(x):
    # Convert to np.ndarray
    x = np.array(x)

    # Getting the dimensions of the array
    dimensions = x.shape

    return dimensions

assert solution([[1,2],[3,4]]) == (2, 2)
```
<SYSTEM>{response}True
<ASSISTANT> {thought} By running this code, you can get the dimensions of an array.
{finish}
```python
import numpy as np

def solution(x):
    # Convert to np.ndarray
    x = np.array(x)

    # Getting the dimensions of the array
    dimensions = x.shape

    return dimensions
```

Begin!
"""

models = [
    dict(
        abbr='gpt-4',
        type=LagentAgent,
        agent_type=ReAct,
        max_turn=3,
        llm=dict(
            type=OpenAI,
            path='gpt-4-0613',
            key='ENV',
            query_per_second=1,
            max_seq_len=4096,
        ),
        actions=[
            dict(
                type=PythonInterpreter,
                answer_expr=None,
                description='''\
It can run a Python code. The code must be a valid code that contains only python method, and the method' name must be 'solution'. The format is:
```python
# import packages
import xxx
def solution(inputs):
    # initialize some variables
    variable_names_with_real_meaning = xxx
    # middle steps
    mid_variable = func(inputs)
    # final answer
    final_answer =  func(mid_variable)
    return final_answer
```'''),
        ],
        protocol=dict(
            type=ReActProtocol,
            call_protocol=FEWSHOT_INSTRUCTION,
            force_stop=FORCE_STOP_PROMPT_EN,
            finish=dict(role='FINISH', begin='Final Answer:', end='\n'),
        ),
        batch_size=8,
        # max_out_len=1024,
        # run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

for dataset in datasets:
    # Use AgentInferencer instead of GenInferencer
    dataset['infer_cfg']['inferencer'] = dict(type=AgentInferencer)
    # # Use the question as agent input directly.
    # dataset['infer_cfg']['prompt_template']['template'] = "{question}"

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=4000),
    runner=dict(
        type=LocalRunner, max_num_workers=16,
        task=dict(type=OpenICLInferTask)),
)
