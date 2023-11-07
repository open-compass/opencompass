import copy
import io
import signal
from contextlib import redirect_stdout
from typing import Any, Optional

from lagent.actions.base_action import BaseAction
from lagent.schema import ActionReturn, ActionStatusCode


class TimeoutError(Exception):
    pass


def handler(signum, frame):
    raise TimeoutError()


class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(
            self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)


DEFAULT_DESCRIPTION = """用来执行Python代码。代码必须是一个函数，
函数名必须得是 'solution'，代码对应你的思考过程。代码实例格式如下：
```python
# import 依赖包
import xxx
def solution():
    # 初始化一些变量
    variable_names_with_real_meaning = xxx
    # 步骤一
    mid_variable = func(variable_names_with_real_meaning)
    # 步骤 x
    mid_variable = func(mid_variable)
    # 最后结果
    final_answer = func(mid_variable)
    return final_answer
```"""


class PythonInterpreter(BaseAction):
    """A Python executor that can execute Python scripts.

    Args:
        description (str): The description of the action. Defaults to
            DEFAULT_DESCRIPTION.
        answer_symbol (str, Optional): the answer symbol from LLM
        answer_expr (str, Optional): the answer function name of the Python
            script. Default to 'solution()'.
        answer_from_stdout (boolean): whether the execution results is from
            stdout.
        name (str, optional): The name of the action. If None, the name will
            be class nameDefaults to None.
        enable (bool, optional): Whether the action is enabled. Defaults to
            True.
        disable_description (str, optional): The description of the action when
            it is disabled. Defaults to None.
        timeout (int): Upper bound of waiting time for Python script execution.
    """

    def __init__(self,
                 description: str = DEFAULT_DESCRIPTION,
                 answer_symbol: Optional[str] = None,
                 answer_expr: Optional[str] = 'solution()',
                 answer_from_stdout: bool = False,
                 name: Optional[str] = None,
                 enable: bool = True,
                 disable_description: Optional[str] = None,
                 timeout: int = 20) -> None:
        super().__init__(description, name, enable, disable_description)

        self.answer_symbol = answer_symbol
        self.answer_expr = answer_expr
        self.answer_from_stdout = answer_from_stdout
        self.timeout = timeout

    def __call__(self, command: str) -> ActionReturn:
        self.runtime = GenericRuntime()
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.timeout)
        try:
            tool_return = self._call(command)
        except TimeoutError as e:
            tool_return = ActionReturn(url=None, args=None, type=self.name)
            tool_return.errmsg = repr(e)
            tool_return.state = ActionStatusCode.API_ERROR
        finally:
            signal.alarm(0)
        return tool_return

    def _call(self, command: str) -> ActionReturn:
        tool_return = ActionReturn(url=None, args=None, type=self.name)
        try:
            if '```python' in command:
                command = command.split('```python')[1].split('```')[0]
            elif '```' in command:
                command = command.split('```')[1].split('```')[0]
            tool_return.args = dict(text='```python\n' + command + '\n```')
            command = command.split('\n')

            if self.answer_from_stdout:
                program_io = io.StringIO()
                with redirect_stdout(program_io):
                    self.runtime.exec_code('\n'.join(command))
                program_io.seek(0)
                res = program_io.readlines()[-1]
            elif self.answer_symbol:
                self.runtime.exec_code('\n'.join(command))
                res = self.runtime._global_vars[self.answer_symbol]
            elif self.answer_expr:
                self.runtime.exec_code('\n'.join(command))
                res = self.runtime.eval_code(self.answer_expr)
            else:
                self.runtime.exec_code('\n'.join(command[:-1]))
                res = True
        except Exception as e:
            tool_return.errmsg = repr(e)
            tool_return.type = self.name
            tool_return.state = ActionStatusCode.API_ERROR
            return tool_return
        try:
            tool_return.result = dict(text=str(res))
            tool_return.state = ActionStatusCode.SUCCESS
        except Exception as e:
            tool_return.errmsg = repr(e)
            tool_return.type = self.name
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return
