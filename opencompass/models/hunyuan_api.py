import json
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

        from tencentcloud.common import credential
        from tencentcloud.common.common_client import CommonClient
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile

        cred = credential.Credential(self.secret_id, self.secret_key)
        httpProfile = HttpProfile()
        httpProfile.endpoint = self.endpoint
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        self.client = CommonClient('hunyuan',
                                   '2023-09-01',
                                   cred,
                                   'ap-beijing',
                                   profile=clientProfile)

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
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            for item in input:
                msg = {'Content': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['Role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['Role'] = 'assistant'
                messages.append(msg)

        from tencentcloud.common.exception.tencent_cloud_sdk_exception import \
            TencentCloudSDKException

        data = {'Messages': messages}

        for _ in range(self.retry):
            try:
                resp = self.client.call_sse('ChatPro', data)
                contents = []
                for event in resp:
                    part = json.loads(event['data'])
                    contents.append(part['Choices'][0]['Delta']['Content'])
                answer = ''.join(contents)

            except TencentCloudSDKException as err:
                print(err)

            print(answer)
            return answer

        raise RuntimeError(f'Failed to respond in {self.retry} retrys')
