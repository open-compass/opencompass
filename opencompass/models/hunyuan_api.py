import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


class Hunyuan(BaseAPIModel):

    def __init__(
        self,
        path: str,
        secret_id: str,
        secret_key: str,
        endpoint: str,
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
    ):  # noqa E125
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            meta_template=meta_template,
            retry=retry,
        )

        self.secret_id = secret_id
        self.secret_key = secret_key
        self.endpoint = endpoint
        self.model = path

        # pip install tencentcloud-sdk-python
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        from tencentcloud.hunyuan.v20230901 import hunyuan_client

        cred = credential.Credential(self.secret_id, self.secret_key)
        httpProfile = HttpProfile()
        httpProfile.endpoint = self.endpoint
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        self.client = hunyuan_client.HunyuanClient(cred, 'ap-beijing',
                                                   clientProfile)

    def generate(self,
                 inputs: List[PromptType],
                 max_out_len: int = 512) -> List[str]:
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

    def _generate(self, input: PromptType, max_out_len: int = 512) -> str:
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

        if isinstance(input, str):
            messages = [{'Role': 'user', 'Content': input}]
        else:
            messages = []
            msg_buffer, last_role = [], None
            for item in input:
                if not item['prompt']:
                    continue
                if item['role'] == 'BOT':
                    role = 'assistant'
                else:  # USER or SYSTEM
                    role = 'user'
                if role != last_role and last_role is not None:
                    messages.append({
                        'Content': '\n'.join(msg_buffer),
                        'Role': last_role
                    })
                    msg_buffer = []
                msg_buffer.append(item['prompt'])
                last_role = role
            messages.append({
                'Content': '\n'.join(msg_buffer),
                'Role': last_role
            })
            messages = messages[-40:]
            if messages[0]['Role'] == 'assistant':
                messages = messages[1:]

        from tencentcloud.common.exception.tencent_cloud_sdk_exception import \
            TencentCloudSDKException
        from tencentcloud.hunyuan.v20230901 import models

        data = {'Model': self.model, 'Messages': messages}

        retry_counter = 0
        while retry_counter < self.retry:
            try:
                req = models.ChatCompletionsRequest()
                req.from_json_string(json.dumps(data))
                resp = self.client.ChatCompletions(req)
                resp = json.loads(resp.to_json_string())
                answer = resp['Choices'][0]['Message']['Content']

            except TencentCloudSDKException as e:
                self.logger.error(f'Got error code: {e.get_code()}')
                if e.get_code() == 'ClientNetworkError':
                    return 'client network error'
                elif e.get_code() in ['InternalError', 'ServerNetworkError']:
                    retry_counter += 1
                    continue
                elif e.get_code() in ['LimitExceeded']:
                    time.sleep(5)
                    continue
                else:
                    print(e)
                    from IPython import embed
                    embed()
                    exit()

            self.logger.debug(f'Generated: {answer}')
            return answer

        raise RuntimeError(f'Failed to respond in {self.retry} retrys')
