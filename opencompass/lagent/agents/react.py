from typing import Union

from lagent.actions import ActionExecutor
from lagent.agents.base_agent import BaseAgent
from lagent.agents.react import ReActProtocol
from lagent.llms.base_api import BaseAPIModel
from lagent.llms.base_llm import BaseModel
from lagent.schema import ActionReturn, AgentReturn


class ReAct(BaseAgent):
    """An implementation of ReAct (https://arxiv.org/abs/2210.03629)

    Args:
        llm (BaseModel or BaseAPIModel): a LLM service which can chat
            and act as backend.
        action_executor (ActionExecutor): an action executor to manage
            all actions and their response.
        protocol (ReActProtocol): a wrapper to generate prompt and
            parse the response from LLM / actions.
        max_turn (int): the maximum number of trails for LLM to generate
            plans that can be successfully parsed by ReWOO protocol.
    """

    def __init__(self,
                 llm: Union[BaseModel, BaseAPIModel],
                 action_executor: ActionExecutor,
                 protocol: ReActProtocol = ReActProtocol(),
                 max_turn: int = 2) -> None:
        self.max_turn = max_turn
        super().__init__(llm=llm,
                         action_executor=action_executor,
                         protocol=protocol)

    def opencompass_adapter(self, prompt):
        # adapter for prompt parsing
        from opencompass.utils.prompt import PromptList
        if isinstance(prompt, list):
            for p in prompt:
                if 'content' in p:
                    p['prompt'] = p.pop('content')
            prompt = PromptList(prompt)
        return prompt

    def chat(self, message: str) -> AgentReturn:
        self._inner_history = []
        self._inner_history.append(dict(role='user', content=message))
        agent_return = AgentReturn()
        force_stop = False
        default_response = '对不起，我无法回答你的问题'
        for turn in range(self.max_turn):
            prompt = self._protocol.format(
                chat_history=self.session_history,
                inner_step=self._inner_history,
                action_executor=self._action_executor,
                force_stop=force_stop)
            prompt = self.opencompass_adapter(prompt)
            # allow single generation
            response = self._llm.generate_from_template([prompt], 512)[0]
            self._inner_history.append(dict(role='assistant',
                                            content=response))
            thought, action, action_input = self._protocol.parse(
                response, self._action_executor)
            action_return: ActionReturn = self._action_executor(
                action, action_input)
            action_return.thought = thought
            agent_return.actions.append(action_return)
            if action_return.type == self._action_executor.finish_action.name:
                agent_return.response = action_return.result['text']
                return agent_return
            self._inner_history.append(
                dict(role='system',
                     content=self._protocol.format_response(action_return)))
            if turn == self.max_turn - 1:
                force_stop = True
        agent_return.response = default_response
        # only append the user and final response
        self._session_history.append(dict(role='user', content=message))
        self._session_history.append(
            dict(role='assistant', content=agent_return.response))
        return agent_return
