import base64
import io
import logging
import os
import queue
import re
import signal
import sys
import traceback
import uuid
from typing import Optional, Tuple

import json5
import PIL.Image
from jupyter_client import KernelManager
from lagent.actions.base_action import BaseAction
from lagent.schema import ActionReturn, ActionStatusCode

WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR',
                     f"{os.path.abspath('./output_images')}")

DEFAULT_DESCRIPTION = """启动Jupter Kernel用于执行Python代码。"""

START_CODE = """
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def input(*args, **kwargs):
    raise NotImplementedError('Python input() function is disabled.')

get_ipython().system = lambda *args: print('Assume we have this package, ! is disabled!')
{}
"""  # noqa


class TimeoutError(Exception):
    pass


class IPythonInterpreter(BaseAction):
    """A IPython executor that can execute Python scripts in a jupyter manner.

    Args:
        description (str): The description of the action. Defaults to
            DEFAULT_DESCRIPTION.
        name (str, optional): The name of the action. If None, the name will
            be class nameDefaults to None.
        enable (bool, optional): Whether the action is enabled. Defaults to
            True.
        disable_description (str, optional): The description of the action when
            it is disabled. Defaults to None.
        timeout (int): Upper bound of waiting time for Python script execution.
            Defaults to 20.
        trim_output (int, optional): Max characters restriction of ipython
            outputs. If None, do not perform any trim.
            Notice that, this is not token length but string length.
            Trim strategies might be added later if needed. Defaults to 1024.
        user_data_dir (str): Specified the user data directory for files
            loading. If set to `ENV`, use `USER_DATA_DIR` environment variable.
            Defaults to `ENV`.
        force_user_data (bool): Whether to force use user data.
            Defaults to True.
    """

    _KERNEL_CLIENTS = {}

    def __init__(self,
                 description: str = DEFAULT_DESCRIPTION,
                 name: Optional[str] = None,
                 enable: bool = True,
                 disable_description: Optional[str] = None,
                 timeout: int = 20,
                 trim_output: Optional[int] = 1024,
                 user_data_dir: str = 'ENV',
                 force_user_data: bool = True) -> None:
        super().__init__(description, name, enable, disable_description)

        self.timeout = timeout
        if user_data_dir == 'ENV':
            user_data_dir = os.environ.get('USER_DATA_DIR', '')

        if user_data_dir:
            # user_data_dir = os.path.dirname(user_data_dir)
            # in case change of dirs
            assert os.path.exists(user_data_dir), \
                f'{user_data_dir} does not exist.'
            user_data_dir = os.path.abspath(user_data_dir)
            user_data_dir = f"import os\nos.chdir('{user_data_dir}')"
        else:
            if force_user_data:
                raise ValueError('user_data_dir is not set. Please '
                                 'set force_user_data to False if '
                                 'no extra data needed.')
        self.user_data_dir = user_data_dir
        self._initialized = False
        self.trim_output = trim_output
        if not os.path.exists(WORK_DIR):
            os.mkdir(WORK_DIR)

    @staticmethod
    def start_kernel():
        # start the kernel and manager
        km = KernelManager()
        km.start_kernel()
        kc = km.client()
        return km, kc

    def initialize(self):
        if self._initialized:
            return
        pid = os.getpid()
        if pid not in self._KERNEL_CLIENTS:
            self._KERNEL_CLIENTS[pid] = self.start_kernel()
        self.kernel_manager, self.kernel_client = self._KERNEL_CLIENTS[pid]
        self._initialized = True
        self._call(START_CODE.format(self.user_data_dir), None)

    def reset(self):
        if not self._initialized:
            self.initialize()
        else:
            code = "get_ipython().run_line_magic('reset', '-f')\n" + \
                START_CODE.format(self.user_data_dir)
            self._call(code, None)

    def _call(self,
              command: str,
              timeout: Optional[int] = None) -> Tuple[str, bool]:
        self.initialize()
        command = extract_code(command)

        # check previous remaining result
        while True:
            try:
                msg = self.kernel_client.get_iopub_msg(timeout=1)
                msg_type = msg['msg_type']
                if msg_type == 'status':
                    if msg['content'].get('execution_state') == 'idle':
                        break
            except queue.Empty:
                # assume no result
                break

        self.kernel_client.execute(command)

        def _inner_call():
            result = ''
            image_path = ''
            succeed = True
            image_idx = 0

            while True:
                text = ''
                image = ''
                finished = False
                msg_type = 'error'
                try:
                    msg = self.kernel_client.get_iopub_msg(timeout=10)
                    msg_type = msg['msg_type']
                    if msg_type == 'status':
                        if msg['content'].get('execution_state') == 'idle':
                            finished = True
                    elif msg_type == 'execute_result':
                        text = msg['content']['data'].get('text/plain', '')
                        if 'image/png' in msg['content']['data']:
                            image_b64 = msg['content']['data']['image/png']
                            image_url = publish_image_to_local(image_b64)
                            image_idx += 1
                            image = '![fig-%03d](%s)' % (image_idx, image_url)
                    elif msg_type == 'display_data':
                        if 'image/png' in msg['content']['data']:
                            image_b64 = msg['content']['data']['image/png']
                            image_url = publish_image_to_local(image_b64)
                            image_idx += 1
                            image = '![fig-%03d](%s)' % (image_idx, image_url)
                        else:
                            text = msg['content']['data'].get('text/plain', '')
                    elif msg_type == 'stream':
                        msg_type = msg['content']['name']  # stdout, stderr
                        text = msg['content']['text']
                    elif msg_type == 'error':
                        succeed = False
                        text = escape_ansi('\n'.join(
                            msg['content']['traceback']))
                        if 'M6_CODE_INTERPRETER_TIMEOUT' in text:
                            text = f'Timeout. No response after {timeout} seconds.'  # noqa
                except queue.Empty:
                    # stop current task in case break next input.
                    self.kernel_manager.interrupt_kernel()
                    succeed = False
                    text = f'Timeout. No response after {timeout} seconds.'
                    finished = True
                except Exception:
                    succeed = False
                    text = 'The code interpreter encountered an unexpected error.'  # noqa
                    logging.warning(''.join(
                        traceback.format_exception(*sys.exc_info())))
                    finished = True
                if text:
                    result += f'\n\n{msg_type}:\n\n```\n{text}\n```'
                if image:
                    image_path += f'\n\n{image}'
                if finished:
                    # in case output text too long
                    # might need better design later
                    if self.trim_output and len(result) > self.trim_output:
                        ellip = '......'
                        half_len = int((self.trim_output - len(ellip)) / 2)
                        result = result[:half_len] + ellip + result[-half_len:]
                    return succeed, result, image_path

        try:
            if timeout:

                def handler(signum, frame):
                    raise TimeoutError()

                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout)
            succeed, result, image_path = _inner_call()
        except TimeoutError:
            succeed = False
            text = 'The code interpreter encountered an unexpected error.'
            result = f'\n\nerror:\n\n```\n{text}\n```'
        finally:
            if timeout:
                signal.alarm(0)

        result = result.lstrip('\n')
        image_path = image_path.lstrip('\n')
        return succeed, result, image_path

    def __call__(self,
                 command: str,
                 timeout: Optional[int] = None) -> ActionReturn:
        tool_return = ActionReturn(url=None, args=None, type=self.name)
        extracted_command = extract_code(command)
        tool_return.args = dict(text=command, extract_code=extracted_command)
        if extracted_command:
            succeed, result, image_path = self._call(extracted_command,
                                                     timeout)
            if succeed:
                if not result:
                    result = 'The code is succeed without any outputs.'
                tool_return.result = dict(text=result, image_path=image_path)
                tool_return.state = ActionStatusCode.SUCCESS
            else:
                tool_return.errmsg = repr(result)
                tool_return.state = ActionStatusCode.API_ERROR
        else:
            tool_return.errmsg = 'The input code is empty. Please follow the format.'  # noqa
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return


def extract_code(text):
    # Match triple backtick blocks first
    triple_match = re.search(r'```[^\n]*\n(.+?)```', text, re.DOTALL)
    # Match single backtick blocks second
    single_match = re.search(r'`([^`]*)`', text, re.DOTALL)
    if triple_match:
        text = triple_match.group(1)
    elif single_match:
        text = single_match.group(1)
    else:
        try:
            text = json5.loads(text)['code']
        except Exception:
            pass
    # If no code blocks found, return original text
    return text


def escape_ansi(line):
    ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', line)


def publish_image_to_local(image_base64: str):
    image_file = str(uuid.uuid4()) + '.png'
    local_image_file = os.path.join(WORK_DIR, image_file)

    png_bytes = base64.b64decode(image_base64)
    assert isinstance(png_bytes, bytes)
    bytes_io = io.BytesIO(png_bytes)
    PIL.Image.open(bytes_io).save(local_image_file, 'png')

    return local_image_file


# local test for code interpreter
def get_multiline_input(hint):
    print(hint)
    print('// Press ENTER to make a new line. Press CTRL-D to end input.')
    lines = []
    while True:
        try:
            line = input()
        except EOFError:  # CTRL-D
            break
        lines.append(line)
    print('// Input received.')
    if lines:
        return '\n'.join(lines)
    else:
        return ''


if __name__ == '__main__':
    code_interpreter = IPythonInterpreter()
    while True:
        print(code_interpreter(get_multiline_input('Enter python code:')))
