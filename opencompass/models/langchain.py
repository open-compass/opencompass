from typing import List, Tuple

from mmengine.registry import Registry

REGISTRY = Registry('helper')


class LangchainAgent:
    """Agent wrapper for Langchain.

    https://github.com/langchain-ai/langchain.
    """

    def __init__(self, agent_type, llm, tools) -> None:
        from langchain.agents import initialize_agent, load_tools

        llm = REGISTRY.build(llm)
        tools = load_tools(tools, llm=llm)
        self.agent = initialize_agent(tools,
                                      llm,
                                      agent=agent_type,
                                      return_intermediate_steps=True)

    def chat(self, user_input, ice=None) -> Tuple[str, List[dict]]:
        from langchain.schema import AgentAction
        try:
            generation = self.agent(user_input)
            answer = generation['output']
            steps = []
            for step in generation['intermediate_steps']:
                action: AgentAction = step[0]
                steps.append(
                    dict(
                        type=action.tool,
                        args=action.tool_input,
                        result=step[1],
                        thought=action.log,
                        state=0,
                        errmsg=None,
                    ))
        except Exception as e:
            answer = None
            steps = [
                dict(
                    type='InvalidAction',
                    args={},
                    result=None,
                    thought=None,
                    state=-1002,
                    errmsg=str(e),
                )
            ]
        return answer, steps
