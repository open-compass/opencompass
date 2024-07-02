# flake8: noqa: E501
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str, float]


class Gemini(BaseAPIModel):
    """Model wrapper around Gemini models.

    Documentation:

    Args:
        path (str): The name of Gemini model.
            e.g. `gemini-pro`
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
        key: str,
        path: str,
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: float = 10.0,
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)
        assert isinstance(key, str)
        if key == 'ENV':
            if 'GEMINI_API_KEY' not in os.environ:
                raise ValueError('GEMINI API key is not set.')
            key = os.getenv('GEMINI_API_KEY')

        assert path in [
            'gemini-1.0-pro', 'gemini-pro', 'gemini-1.5-flash',
            'gemini-1.5-pro'
        ]  # https://ai.google.dev/gemini-api/docs/models/gemini#model-variations

        self.url = f'https://generativelanguage.googleapis.com/v1beta/models/{path}:generateContent?key={key}'
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.headers = {
            'content-type': 'application/json',
        }

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
    ) -> str:
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
            messages = [{'role': 'user', 'parts': [{'text': input}]}]
        else:
            messages = []
            system_prompt = None
            for item in input:
                if item['role'] == 'SYSTEM':
                    system_prompt = item['prompt']
            for item in input:
                if system_prompt is not None:
                    msg = {
                        'parts': [{
                            'text': system_prompt + '\n' + item['prompt']
                        }]
                    }
                else:
                    msg = {'parts': [{'text': item['prompt']}]}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                    messages.append(msg)
                elif item['role'] == 'BOT':
                    msg['role'] = 'model'
                    messages.append(msg)
                elif item['role'] == 'SYSTEM':
                    pass

            # model can be response with user and system
            # when it comes with agent involved.
            assert msg['role'] in ['user', 'system']

        data = {
            'model':
            self.path,
            'contents':
            messages,
            'safetySettings': [
                {
                    'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                    'threshold': 'BLOCK_NONE'
                },
                {
                    'category': 'HARM_CATEGORY_HATE_SPEECH',
                    'threshold': 'BLOCK_NONE'
                },
                {
                    'category': 'HARM_CATEGORY_HARASSMENT',
                    'threshold': 'BLOCK_NONE'
                },
                {
                    'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                    'threshold': 'BLOCK_NONE'
                },
            ],
            'generationConfig': {
                'candidate_count': 1,
                'temperature': self.temperature,
                'maxOutputTokens': 2048,
                'topP': self.top_p,
                'topK': self.top_k
            }
        }

        for _ in range(self.retry):
            self.wait()
            raw_response = requests.post(self.url,
                                         headers=self.headers,
                                         data=json.dumps(data))
            try:
                response = raw_response.json()
            except requests.JSONDecodeError:
                self.logger.error('JsonDecode error, got',
                                  str(raw_response.content))
                time.sleep(1)
                continue
            if raw_response.status_code == 200:
                if 'candidates' not in response:
                    self.logger.error(response)
                else:
                    if 'content' not in response['candidates'][0]:
                        return "Due to Google's restrictive policies, I am unable to respond to this question."
                    else:
                        return response['candidates'][0]['content']['parts'][
                            0]['text'].strip()
            try:
                msg = response['error']['message']
                self.logger.error(msg)
            except KeyError:
                pass
            self.logger.error(response)
            time.sleep(1)

        raise RuntimeError('API call failed.')
