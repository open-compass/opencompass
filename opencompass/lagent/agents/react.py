import re
from typing import Union

from lagent.actions import ActionExecutor
from lagent.agents.base_agent import BaseAgent
from lagent.agents.react import ReActProtocol
from lagent.llms.base_api import BaseAPIModel
from lagent.llms.base_llm import BaseModel
from lagent.schema import ActionReturn, ActionStatusCode, AgentReturn


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

    def reset(self):
        """Reset history."""
        self._session_history = []

    def opencompass_adapter(self, prompt):
        # adapter for prompt parsing
        if isinstance(prompt, list):
            system_prompt = []
            merged_prompt = []
            for p in prompt:
                tmp_p = p.copy()
                if 'content' in tmp_p:
                    tmp_p['prompt'] = tmp_p.pop('content')
                if 'role' in tmp_p:
                    if tmp_p['role'] == 'system':
                        # skip system prompt
                        system_prompt.append(tmp_p['prompt'])
                        continue
                    # no system for meta template temperaily
                    if tmp_p['role'] == 'assistant':
                        tmp_p['role'] = 'BOT'
                    if tmp_p['role'] == 'user':
                        # merge previous system prompt to user
                        system_str = ''.join(system_prompt)
                        tmp_p['prompt'] = system_str + tmp_p['prompt']
                        tmp_p['role'] = 'HUMAN'
                        system_prompt = []
                merged_prompt.append(tmp_p)

            # merge if system still exists
            if system_prompt:
                if 'role' in merged_prompt[-1]:
                    if merged_prompt[-1]['role'] == 'HUMAN':
                        # append to the final human prompt
                        merged_prompt[-1]['prompt'] += ''.join(system_prompt)
                    else:
                        # create a human prompt behind
                        merged_prompt.append(
                            dict(role='HUMAN', prompt=''.join(system_prompt)))

        from opencompass.utils.prompt import PromptList
        new_prompt = PromptList()
        # adapter for meta template
        new_prompt.append(dict(section='round', pos='begin'))
        new_prompt.extend(merged_prompt)
        new_prompt.append(dict(section='round', pos='end'))

        return new_prompt

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

            # TODO: hard code here
            action_input = re.sub('<eoa>', '', action_input)

            if 'tensorflow' in action_input:
                # skip tensorflow currently
                break
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


class CIReAct(ReAct):
    """Code Interpreter version of ReAct. The success state is different from
    ReAct.

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

    def reset(self):
        """Reset history and reset action if suit the case."""
        self._session_history = []
        # hard code here
        from opencompass.lagent.actions.ipython_interpreter import \
            IPythonInterpreter
        b = IPythonInterpreter()
        b.reset()

    def chat(self, message: str) -> AgentReturn:
        self._inner_history = []
        # append the user message for session history
        self._session_history.append(dict(role='user', content=message))
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
            if action_return.state == ActionStatusCode.SUCCESS:
                # if success, stash model response and system response
                self._session_history.append(
                    dict(role='assistant', content=action_return.args['text']))
                self._session_history.append(
                    dict(
                        role='system',
                        content=self._protocol.format_response(action_return)))
                agent_return.response = action_return.result['text']
                return agent_return
            elif action_return.type == self._action_executor.invalid_action.name:  # noqa
                action_return.errmsg = 'The action is invalid, please check the action name.'  # noqa
            self._inner_history.append(
                dict(role='system',
                     content=self._protocol.format_response(action_return)))
            if turn == self.max_turn - 1:
                force_stop = True
        agent_return.response = default_response
        self._session_history.append(
            dict(role='assistant', content=agent_return.response))
        return agent_return
