from copy import deepcopy
from typing import List, Tuple

from mmengine.registry import Registry

REGISTRY = Registry('helper')


class LagentAgent:
    """Agent wrapper for Lagent.

    https://github.com/InternLM/lagent.
    """
    is_api = True

    def __init__(self, agent_type, llm, actions=None, protocol=None, **kwargs):
        llm = REGISTRY.build(llm)
        agent_cfg = {'type': agent_type, 'llm': llm, **kwargs}

        if actions is not None:
            from lagent.actions import ActionExecutor
            executor = ActionExecutor([])
            for action in actions:
                action = REGISTRY.build(action)
                if 'agentlego' in type(action).__module__:
                    action = action.to_lagent()
                executor.add_action(action)
            agent_cfg['action_executor'] = executor
        if protocol is not None:
            protocol = REGISTRY.build(protocol)
            agent_cfg['protocol'] = protocol

        from lagent import BaseAgent
        self.agent: BaseAgent = REGISTRY.build(agent_cfg)

    def reset(self):
        self.agent._session_history = []
        for action in self.agent._action_executor.actions:
            if hasattr(action, 'reset'):
                action.reset()

    def set_history(self, history):
        self.agent._session_history = deepcopy(history)

    def gt_response(self, prompt):
        if 'CIReAct' in str(self.agent.__class__):
            thought, gold = prompt.split('**split**')
            prompt = f"""{self.agent._protocol.thought['begin']} {thought}\
\n{self.agent._protocol.action['begin']} IPythonInterpreter\n\
{self.agent._protocol.action_input['begin']}```python\n{gold}\n```\n"""  # noqa
            action_input = dict(
                command=f"""```python\n{gold}\n```\n""",
                timeout=120,
            )
            response = self.agent._action_executor('IPythonInterpreter',
                                                   action_input)
            gt_response = dict(role='assistant', content=prompt)
            system_response = dict(
                role='system',
                content=self.agent._protocol.format_response(response))
            return [gt_response, system_response]
        else:
            gt_response = dict(role='assistant', content=prompt)
            return [gt_response]

    @property
    def template_parser(self):
        return self.agent._llm.template_parser

    @template_parser.setter
    def template_parser(self, value):
        self.agent._llm.template_parser = value

    def chat(self,
             user_input: str,
             history: List[dict] = None) -> Tuple[str, List[dict], List[dict]]:
        """Chat with agent."""
        if history:
            self.agent._session_history = history

        from lagent.schema import ActionReturn, AgentReturn
        generation: AgentReturn = self.agent.chat(user_input)

        inner_steps = generation.inner_steps
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

        return answer, steps, inner_steps


FORCE_STOP_PROMPT_EN = (
    """You should directly give results based on history information."""  # noqa
)

FEWSHOT_INSTRUCTION = """\
You are a assistant who can utilize external tools.
{tool_description}
To use a tool, please response with the following format:
```
{thought} Think what you need to solve, do you need to use tools?
{action} The tool name, should be one of [{action_names}].
{action_input} The input to the tool that you want to use.
```
The tool will give you response after your response using the following format:
```
{response} the results after call the tool.
```
Therefore DO NOT generate tool response by yourself.

Also please follow the guidelines:
1. Always use code interpreter to solve the problem.
2. The generated codes should always in a markdown code block format.
3. The generated codes will be executed in an ipython manner and the results will be cached.
4. Your responded code should always be simple and only solves the problem in current step.

Begin!
"""  # noqa

PYTHON_INTERPRETER_DESCRIPTION = """\
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
    final_answer = func(mid_variable)
    return final_answer
```"""  # noqa


class CodeAgent(LagentAgent):
    """Code Agent wrapper for Lagent."""

    def __init__(self, llm, **kwargs):
        from lagent import PythonInterpreter, ReAct
        from lagent.agents.react import ReActProtocol

        agent_type = kwargs.pop('agent_type', ReAct)
        max_turn = kwargs.pop('max_turn', 3)
        actions = kwargs.pop(
            'actions',
            [
                dict(type=PythonInterpreter,
                     description=PYTHON_INTERPRETER_DESCRIPTION),
            ],
        )
        protocol = kwargs.pop(
            'protocol',
            dict(
                type=ReActProtocol,
                call_protocol=FEWSHOT_INSTRUCTION,
                force_stop=FORCE_STOP_PROMPT_EN,
                finish=dict(role='FINISH', begin='Final Answer:', end='\n'),
            ),
        )
        super().__init__(agent_type=agent_type,
                         llm=llm,
                         actions=actions,
                         protocol=protocol,
                         max_turn=max_turn,
                         **kwargs)
