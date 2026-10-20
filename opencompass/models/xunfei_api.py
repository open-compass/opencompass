import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


class XunFei(BaseAPIModel):
    """Model wrapper around XunFei.

    Args:
        path (str): Provided URL.
        appid (str): Provided APPID.
        api_secret (str): Provided APISecret.
        api_key (str): Provided APIKey.
        domain (str): Target version domain. Defaults to `general`.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 2.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    def __init__(self,
                 path: str,
                 appid: str,
                 api_secret: str,
                 api_key: str,
                 domain: str = 'general',
                 query_per_second: int = 2,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 retry: int = 2):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)
        import ssl
        import threading
        from urllib.parse import urlencode, urlparse

        import websocket
        self.urlencode = urlencode
        self.websocket = websocket
        self.websocket.enableTrace(False)
        self.threading = threading
        self.ssl = ssl

        # weird auth keys
        self.APISecret = api_secret
        self.APIKey = api_key
        self.domain = domain
        self.appid = appid
        self.hostname = urlparse(path).netloc
        self.hostpath = urlparse(path).path

        self.headers = {
            'content-type': 'application/json',
        }

    def get_url(self):
        from datetime import datetime
        from time import mktime
        from wsgiref.handlers import format_date_time

        cur_time = datetime.now()
        date = format_date_time(mktime(cur_time.timetuple()))
        tmp = f'host: {self.hostname}\n'
        tmp += 'date: ' + date + '\n'
        tmp += 'GET ' + self.hostpath + ' HTTP/1.1'
        import hashlib
        import hmac
        tmp_sha = hmac.new(self.APISecret.encode('utf-8'),
                           tmp.encode('utf-8'),
                           digestmod=hashlib.sha256).digest()
        import base64
        signature = base64.b64encode(tmp_sha).decode(encoding='utf-8')
        authorization_origin = (f'api_key="{self.APIKey}", '
                                'algorithm="hmac-sha256", '
                                'headers="host date request-line", '
                                f'signature="{signature}"')
        authorization = base64.b64encode(
            authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        v = {
            'authorization': authorization,
            'date': date,
            'host': self.hostname
        }
        url = self.path + '?' + self.urlencode(v)
        return url

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs)))
        self.flush()
        return results

    def _generate(
        self,
        input: PromptType,
        max_out_len: int = 512,
    ) -> List[str]:
        """Generate results given an input.

        Args:
            inputs (PromptType): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))

        # FIXME: messages only contains the last input
        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            # word_ctr = 0
            # TODO: Implement truncation in PromptList
            for item in input:
                msg = {'content': item['prompt']}
                # if word_ctr >= self.max_seq_len:
                #     break
                # if len(msg['content']) + word_ctr > self.max_seq_len:
                #     msg['content'] = msg['content'][word_ctr -
                #                                     self.max_seq_len:]
                # word_ctr += len(msg['content'])
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['role'] = 'assistant'
                messages.append(msg)
            # in case the word break results in even number of messages
            # if len(messages) > 0 and len(messages) % 2 == 0:
            #     messages = messages[:-1]

        data = {
            'header': {
                'app_id': self.appid,
            },
            'parameter': {
                'chat': {
                    'domain': self.domain,
                    'max_tokens': max_out_len,
                }
            },
            'payload': {
                'message': {
                    'text': messages
                }
            }
        }

        msg = ''
        err_code = None
        err_data = None
        content_received = self.threading.Event()

        def on_open(ws):
            nonlocal data
            ws.send(json.dumps(data))

        def on_message(ws, message):
            nonlocal msg, err_code, err_data, content_received
            err_data = json.loads(message)
            err_code = err_data['header']['code']
            if err_code != 0:
                content_received.set()
                ws.close()
            else:
                choices = err_data['payload']['choices']
                status = choices['status']
                msg += choices['text'][0]['content']
                if status == 2:
                    content_received.set()
                    ws.close()

        ws = self.websocket.WebSocketApp(self.get_url(),
                                         on_message=on_message,
                                         on_open=on_open)
        ws.appid = self.appid
        ws.question = messages[-1]['content']

        for _ in range(self.retry):
            self.acquire()
            ws.run_forever(sslopt={'cert_reqs': self.ssl.CERT_NONE})
            content_received.wait()
            self.release()
            if err_code == 0:
                return msg.strip()
            if err_code == 10014:  # skip safety problem
                return 'None'

        if err_code == 10013:
            return err_data['header']['message']
        raise RuntimeError(f'Code: {err_code}, data: {err_data}')


class XunFeiSpark(BaseAPIModel):
    """Model wrapper around XunFeiSpark.

    Documentation:

    Args:
        path (str): The name of XunFeiSpark model.
            e.g. `moonshot-v1-32k`
        key (str): Authorization key.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    def __init__(
        self,
        path: str,
        url: str,
        app_id: str,
        api_key: str,
        api_secret: str,
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)
        try:
            from sparkai.llm.llm import ChatSparkLLM  # noqa: F401
        except ImportError:
            raise ImportError('run `pip install --upgrade spark_ai_python`')

        self.spark_domain = path
        self.url = url
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
    ) -> List[str]:
        results = [self._generate(input, max_out_len) for input in inputs]
        return results

    def _generate(
        self,
        input: PromptType,
        max_out_len: int = 512,
    ) -> str:
        assert isinstance(input, (str, PromptList))

        from sparkai.core.messages import ChatMessage
        from sparkai.errors import SparkAIConnectionError
        from sparkai.llm.llm import ChatSparkLLM

        if isinstance(input, str):
            messages = [ChatMessage(role='user', content=input)]
        else:
            messages = []
            msg_buffer, last_role = [], None
            for index, item in enumerate(input):
                if index == 0 and item['role'] == 'SYSTEM':
                    role = 'system'
                elif item['role'] == 'BOT':
                    role = 'assistant'
                else:
                    role = 'user'

                if role != last_role and last_role is not None:
                    content = '\n'.join(msg_buffer)
                    messages.append(
                        ChatMessage(role=last_role, content=content))
                    msg_buffer = []

                msg_buffer.append(item['prompt'])
                last_role = role

            content = '\n'.join(msg_buffer)
            messages.append(ChatMessage(role=last_role, content=content))

        spark = ChatSparkLLM(
            spark_api_url=self.url,
            spark_app_id=self.app_id,
            spark_api_key=self.api_key,
            spark_api_secret=self.api_secret,
            spark_llm_domain=self.spark_domain,
            streaming=False,
            max_tokens=max_out_len,
        )

        all_empty_response = True
        for _ in range(self.retry + 1):
            try:
                outputs = spark.generate([messages]).generations[0]
                if len(outputs) == 0:
                    self.logger.error('Empty response, retrying...')
                    continue
                msg = outputs[0].text
                self.logger.debug(f'Generated: {msg}')
                return msg
            except (ConnectionError, SparkAIConnectionError) as e:
                if isinstance(e, SparkAIConnectionError):
                    error_code = e.error_code
                    message = e.message
                else:
                    match = re.match(r'Error Code: (\d+), Error: (.*)',
                                     e.args[0],
                                     flags=re.DOTALL)
                    if not match:
                        raise e
                    error_code = int(match.group(1))
                    message = match.group(2)

                if error_code == 10003:  # query data exceed limit
                    self.logger.error(f'Error {error_code}: {message}')
                    return message
                elif error_code in [10013, 10014]:  # skip safety problem
                    self.logger.debug(f'Generated: {message}')
                    return message
                elif error_code == 10020:  # plugin result is empty
                    self.logger.error(f'Error {error_code}: {message}')
                    return message
                elif error_code == 11202:  # qps limit
                    time.sleep(1)
                    continue
                else:
                    self.logger.error(f'Error {error_code}: {message}')
                    raise e
            except TimeoutError:
                self.logger.error('TimeoutError, sleep 60, retrying...')
                time.sleep(60)
            except Exception as e:
                self.logger.error(str(e))
                pass

            all_empty_response = False

        if all_empty_response:
            self.logger.error('All empty response')
            return 'all empty response'

        raise RuntimeError('Failed to generate response')
