from typing import List, Tuple

from mmengine.registry import Registry

REGISTRY = Registry('helper')


class LagentAgent:
    """Agent wrapper for Lagent.

    https://github.com/InternLM/lagent.
    """

    def __init__(self, agent_type, llm, actions=None, protocol=None, **kwargs):
        llm = REGISTRY.build(llm)
        # protocol illustration example
        example = kwargs.pop('example', '')
        agent_cfg = {'type': agent_type, 'llm': llm, **kwargs}

        if actions is not None:
            from lagent.actions import ActionExecutor
            executor = ActionExecutor(
                [REGISTRY.build(action) for action in actions])
            agent_cfg['action_executor'] = executor
        if protocol is not None:
            # format example in protocol if needed
            if '{example}' in protocol['call_protocol']:
                protocol['call_protocol'] = protocol['call_protocol'].format(
                    example=example)
            protocol = REGISTRY.build(protocol)
            agent_cfg['protocol'] = protocol

        self.agent = REGISTRY.build(agent_cfg)

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
