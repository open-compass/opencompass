import json
import os
import time

import numpy as np
import openai
import requests

OPENICL_API_NAME_LIST = ['opt-175b', 'gpt3']
OPENICL_API_PARAMETER_DICT = {
    'opt-175b': ['URL', 'headers'],
    'gpt3': [
        'engine', 'temperature', 'max_tokens', 'top_p', 'frequency_penalty',
        'presence_penalty', 'sleep_time'
    ]
}
OPENICL_API_REQUEST_CONFIG = {
    'opt-175b': {
        'URL': '',  # http://xxx/completions or http://xxx/generate
        'headers': {
            'Content-Type': 'application/json; charset=UTF-8'
        }
    },
    'gpt3': {
        'engine': 'text-davinci-003',
        'temperature': 0,
        'max_tokens': 256,
        'top_p': 1.0,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0,
        'sleep_time': 3
    }
}
PROXIES = {'https': '', 'http': ''}


def is_api_available(api_name):
    if api_name is None:
        return False
    return True if api_name in OPENICL_API_NAME_LIST else False


def update_openicl_api_request_config(api_name, **kwargs):
    if api_name is None or not is_api_available(api_name):
        return

    parameter_list = OPENICL_API_PARAMETER_DICT[api_name]
    for parameter in parameter_list:
        if parameter in kwargs.keys():
            OPENICL_API_REQUEST_CONFIG[api_name][parameter] = kwargs[parameter]


def api_get_ppl(api_name, input_texts):
    if api_name == 'opt-175b':
        pyload = {'prompt': input_texts, 'max_tokens': 0, 'echo': True}
        response = json.loads(
            requests.post(
                OPENICL_API_REQUEST_CONFIG[api_name]['URL'],
                data=json.dumps(pyload),
                headers=OPENICL_API_REQUEST_CONFIG[api_name]['headers'],
                proxies=PROXIES).text)
        lens = np.array(
            [len(r['logprobs']['tokens']) for r in response['choices']])
        ce_loss = np.array([
            -sum(r['logprobs']['token_logprobs']) for r in response['choices']
        ])
        return ce_loss / lens

    if api_name == 'gpt3':
        raise NotImplementedError("GPT-3 API doesn't support PPL calculation")


def api_get_tokens(api_name, input_texts):
    length_list = [len(text) for text in input_texts]

    if api_name == 'opt-175b':
        pyload = {'prompt': input_texts, 'max_tokens': 100, 'echo': True}
        response = json.loads(
            requests.post(
                OPENICL_API_REQUEST_CONFIG[api_name]['URL'],
                data=json.dumps(pyload),
                headers=OPENICL_API_REQUEST_CONFIG[api_name]['headers'],
                proxies=PROXIES).text)
        return [r['text'] for r in response['choices']], [
            r['text'][length:]
            for r, length in zip(response['choices'], length_list)
        ]

    if api_name == 'gpt3':
        openai.api_key = os.getenv('OPENAI_API_KEY')
        response = openai.Completion.create(
            engine=OPENICL_API_REQUEST_CONFIG['gpt3']['engine'],
            prompt=input_texts,
            temperature=OPENICL_API_REQUEST_CONFIG['gpt3']['temperature'],
            max_tokens=OPENICL_API_REQUEST_CONFIG['gpt3']['max_tokens'],
            top_p=OPENICL_API_REQUEST_CONFIG['gpt3']['top_p'],
            frequency_penalty=OPENICL_API_REQUEST_CONFIG['gpt3']
            ['frequency_penalty'],
            presence_penalty=OPENICL_API_REQUEST_CONFIG['gpt3']
            ['presence_penalty'])
        time.sleep(OPENICL_API_REQUEST_CONFIG['gpt3']['sleep_time'])
        return [(input + r['text'])
                for r, input in zip(response['choices'], input_texts)
                ], [r['text'] for r in response['choices']]
