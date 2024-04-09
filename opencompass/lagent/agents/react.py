import copy
from typing import Dict, List

from lagent.actions import ActionExecutor
from lagent.agents.react import ReAct as _ReAct
from lagent.agents.react import ReActProtocol as _ReActProtocol
from lagent.schema import ActionReturn, ActionStatusCode, AgentReturn


class ReActProtocol(_ReActProtocol):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # defaults to system
        self.system_role = 'system'
        self.first_system_role = 'system'
        self.merge_adjacent_role = False

    def format(self,
               chat_history: List[Dict],
               inner_step: List[Dict],
               action_executor: ActionExecutor,
               force_stop: bool = False) -> list:
        """Generate the ReAct format prompt.

        Args:
            chat_history (List[Dict]): The history log in previous runs.
            inner_step (List[Dict]): The log in the current run.
            action_executor (ActionExecutor): the action manager to
                execute actions.
            force_stop (boolean): whether force the agent to give responses
                under pre-defined turns.

        Returns:
            List[Dict]: ReAct format prompt.
        """

        call_protocol = self.call_protocol.format(
            tool_description=action_executor.get_actions_info(),
            action_names=action_executor.action_names(),
            thought=self.thought['begin'],
            action=self.action['begin'],
            action_input=self.action_input['begin'],
            response=self.response['begin'],
            finish=self.finish['begin'],
        )
        formatted = []
        formatted.append(
            dict(role=self.first_system_role, content=call_protocol))
        formatted += chat_history
        formatted += inner_step
        if force_stop:
            formatted.append(
                dict(role=self.system_role, content=self.force_stop))

        if self.merge_adjacent_role and formatted:
            merged = [formatted[0]]  # Add the first dict

            for d in formatted[1:]:
                # If the 'role' of current dict matches with the 'role' of the
                # last dict in merged list,
                # append its 'content' to the 'content' of the last dict.
                if d['role'] == merged[-1]['role']:
                    merged[-1]['content'] += d['content']
                else:
                    # If 'role' does not match, add it as a new dict in the
                    # merged list
                    merged.append(d)

            return merged

        return formatted


class ReAct(_ReAct):

    def __init__(self,
                 use_system_role: bool = True,
                 first_system_role: bool = True,
                 merge_adjacent_role: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if use_system_role:
            self.system_role = 'system'
        else:
            self.system_role = 'user'
        if use_system_role or first_system_role:
            first_system_role = 'system'
        else:
            first_system_role = 'user'
        self._protocol.first_system_role = first_system_role
        self._protocol.system_role = self.system_role
        self._protocol.merge_adjacent_role = merge_adjacent_role

    def chat(self, message: str) -> AgentReturn:
        for hist in self._session_history:
            if hist['role'] == 'system':
                hist['role'] = self.system_role
        self._inner_history = []
        self._inner_history.append(dict(role='user', content=message))
        agent_return = AgentReturn()
        default_response = 'Sorry that I cannot answer your question.'
        for turn in range(self.max_turn):
            prompt = self._protocol.format(
                chat_history=self.session_history,
                inner_step=self._inner_history,
                action_executor=self._action_executor,
                force_stop=(turn == self.max_turn - 1))
            response = self._llm.generate_from_template(prompt, 512)
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
                break
            self._inner_history.append(
                dict(role=self.system_role,
                     content=self._protocol.format_response(action_return)))
        else:
            agent_return.response = default_response
        agent_return.inner_steps = copy.deepcopy(self._inner_history)
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
        for hist in self._session_history:
            if hist['role'] == 'system':
                hist['role'] = self.system_role
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
            response = self._llm.generate_from_template(prompt, 512)
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
                    dict(role='assistant', content=response))
                self._session_history.append(
                    dict(
                        role=self.system_role,
                        content=self._protocol.format_response(action_return)))
                agent_return.response = action_return.result['text']
                return agent_return
            elif action_return.type == self._action_executor.invalid_action.name:  # noqa
                action_return.errmsg = 'The action is invalid, please check the action name.'  # noqa
            self._inner_history.append(
                dict(role=self.system_role,
                     content=self._protocol.format_response(action_return)))
            if turn == self.max_turn - 1:
                force_stop = True
        agent_return.response = default_response
        self._session_history.append(
            dict(role='assistant', content=agent_return.response))
        return agent_return


class CIReActMergeRole(CIReAct):
    """如有第一轮 SYSTEM, 则使用 SYSTEM。后续 SYSTEM 使用 USER 合并复数轮 USER USER 与 BOT
    交替出现."""

    def chat(self, message: str) -> AgentReturn:
        for hist in self._session_history:
            if hist['role'] == 'system':
                hist['role'] = self.system_role
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
            prompt = self.merge_role(prompt)
            response = self._llm.generate_from_template(prompt, 512)
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
                    dict(role='assistant', content=response))
                self._session_history.append(
                    dict(
                        role=self.system_role,
                        content=self._protocol.format_response(action_return)))
                agent_return.response = action_return.result['text']
                return agent_return
            elif action_return.type == self._action_executor.invalid_action.name:  # noqa
                action_return.errmsg = 'The action is invalid, please check the action name.'  # noqa
            self._inner_history.append(
                dict(role=self.system_role,
                     content=self._protocol.format_response(action_return)))
            if turn == self.max_turn - 1:
                force_stop = True
        agent_return.response = default_response
        self._session_history.append(
            dict(role='assistant', content=agent_return.response))
        return agent_return

    def merge_role(self, inputs):
        messages = []
        msg_buffer, last_role = [], None
        for index, item in enumerate(inputs):
            if index == 0 and item['role'] == 'system':
                role = 'system'
            elif item['role'] == 'assistant':
                role = 'assistant'
            else:
                role = 'user'
            if role != last_role and last_role is not None:
                messages.append({
                    'content': '\n'.join(msg_buffer),
                    'role': last_role
                })
                msg_buffer = []
            msg_buffer.append(item['content'])
            last_role = role
        messages.append({'content': '\n'.join(msg_buffer), 'role': last_role})
        return messages
