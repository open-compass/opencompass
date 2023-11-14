from typing import List, Tuple

from mmengine.registry import Registry

from opencompass.lagent.agents.react import ReAct
from opencompass.utils import get_logger

REGISTRY = Registry('helper')


class LagentAgent:
    """Agent wrapper for Lagent.

    https://github.com/InternLM/lagent.
    """

    def __init__(self, agent_type, llm, actions=None, protocol=None, **kwargs):
        llm = REGISTRY.build(llm)
        agent_cfg = {'type': agent_type, 'llm': llm, **kwargs}

        if actions is not None:
            from lagent.actions import ActionExecutor
            executor = ActionExecutor(
                [REGISTRY.build(action) for action in actions])
            agent_cfg['action_executor'] = executor
        if protocol is not None:
            protocol = REGISTRY.build(protocol)
            agent_cfg['protocol'] = protocol

        self.agent = REGISTRY.build(agent_cfg)

    def add_example(self, example):
        # format example in protocol if needed
        call_protocol = self.agent._protocol.call_protocol
        if '{example}' in call_protocol:
            self.agent._protocol.call_protocol = call_protocol.format(
                example=example)
        else:
            get_logger().warning('Protocal template does not have example'
                                 ' placeholder, please check your template.')

    def chat(self, user_input, ice=None) -> Tuple[str, List[dict]]:
        from lagent.schema import ActionReturn, AgentReturn
        generation: AgentReturn = self.agent.chat(user_input)
        self.agent._session_history = []  # clear agent history
        answer = generation.response
        steps = []

        for step in generation.actions:
            step: ActionReturn
            steps.append(
                dict(
                    type=step.type,
                    args=step.args,
                    result=step.result,
                    thought=step.thought,
                    state=int(step.state),
                    errmsg=step.errmsg,
                    valid=int(step.valid),
                ))
        return answer, steps


FORCE_STOP_PROMPT_EN = """You should directly give results based on history information."""  # noqa

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
""" # noqa

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
```''' # noqa


class CodeAgent:
    """Agent wrapper for Lagent."""

    def __new__(self, llm, **kwargs):
        from lagent.actions import PythonInterpreter
        from lagent.agents.react import ReActProtocol
        agent_type = kwargs.pop('agent_type', ReAct)
        max_turn = kwargs.pop('max_turn', 3)
        actions = kwargs.pop('actions', [
            dict(type=PythonInterpreter,
                 description=PYTHON_INTERPRETER_DESCRIPTION),
        ])
        protocol = kwargs.pop(
            'protocol',
            dict(
                type=ReActProtocol,
                call_protocol=FEWSHOT_INSTRUCTION,
                force_stop=FORCE_STOP_PROMPT_EN,
                finish=dict(role='FINISH', begin='Final Answer:', end='\n'),
            ))
        return LagentAgent(agent_type=agent_type,
                           llm=llm,
                           max_turn=max_turn,
                           actions=actions,
                           protocol=protocol,
                           **kwargs)
