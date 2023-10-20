from opencompass.lagent.agents.react import ReAct
from opencompass.models.lagent import LagentAgent
from lagent.agents.react import ReActProtocol
from lagent.actions import PythonInterpreter

FORCE_STOP_PROMPT_EN = """You should directly give results based on history information."""

FEWSHOT_INSTRUCTION = """\
You are a assistant who can utilize external tools.
{{tool_description}}
To use a tool, please use the following format:
```
{{thought}} Think what you need to solve, do you need to use tools?
{{action}} the tool name, should be one of [{{action_names}}]
{{action_input}} the input to the action
```
I will give you response after utilizing tools should using the following format:
```
{{response}} the results after call the tool.
``
If you already know the answer, or you do not need to use tools,
please using the following format to reply:
```
{{thought}} the thought process to get the final answer
{{finish}} final answer
```

Example:
{example}

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

# need model config
model_template = dict(
    type=LagentAgent,
    agent_type=ReAct,
    max_turn=3,
    actions=[
        dict(
            type=PythonInterpreter,
            description=PYTHON_INTERPRETER_DESCRIPTION),
    ],
    protocol=dict(
        type=ReActProtocol,
        call_protocol=FEWSHOT_INSTRUCTION,
        force_stop=FORCE_STOP_PROMPT_EN,
        finish=dict(role='FINISH', begin='Final Answer:', end='\n'),
    ))
