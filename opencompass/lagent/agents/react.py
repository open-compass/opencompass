from lagent.agents.react import ReAct
from lagent.schema import ActionReturn, ActionStatusCode, AgentReturn


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
