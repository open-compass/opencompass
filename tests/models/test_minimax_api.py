import copy
import json
import os
import unittest
from unittest.mock import MagicMock, patch

import requests

from opencompass.configs.models.minimax.minimax_m3 import models as m3_models
from opencompass.configs.models.minimax.minimax_m3_1_flash import \
    models as flash_models
from opencompass.models.minimax_api import MiniMaxChatCompletionV2
from opencompass.utils.build import build_model_from_cfg
from opencompass.utils.prompt import PromptList


class TestMiniMaxChatCompletionV2(unittest.TestCase):

    def _build_model(self, **kwargs):
        model = MiniMaxChatCompletionV2(**kwargs)
        model.tokens = MagicMock()
        return model

    def _generate(self, model, prompt='hello', max_out_len=512):
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({
            'choices': [{'message': {'content': 'answer'}}]
        }).encode()
        # Inspect the serialized request after Requests prepares it, without
        # sending credentials or prompts to a real endpoint.
        with patch('requests.sessions.Session.send',
                   return_value=response) as send:
            result = model.generate([prompt], max_out_len=max_out_len)
        self.assertEqual(result, ['answer'])
        send.assert_called_once()
        prepared = send.call_args.args[0]
        self.assertEqual(prepared.method, 'POST')
        return prepared, json.loads(prepared.body)

    @patch.dict(os.environ, {'MINIMAX_API_KEY': 'test-key'})
    def test_default_flash_request(self):
        model = self._build_model()
        prepared, payload = self._generate(model)
        self.assertEqual(model.max_seq_len, 1000000)
        self.assertEqual(prepared.url,
                         'https://api.minimax.io/v1/chat/completions')
        self.assertEqual(prepared.headers['Authorization'], 'Bearer test-key')
        self.assertEqual(payload, {
            'model': 'MiniMax-M3.1-Flash',
            'messages': [{'role': 'user', 'content': 'hello'}],
            'max_tokens': 512,
        })
        # Omission keeps the API's adaptive thinking and max effort defaults;
        # it also avoids unsupported service_tier and reasoning_split=False.

    @patch.dict(os.environ, {'MINIMAX_API_KEY': 'test-key'})
    def test_current_configs_reach_transport(self):
        for configs, name, output_limit in [
            (flash_models, 'MiniMax-M3.1-Flash', 524288),
            (m3_models, 'MiniMax-M3', 131072),
        ]:
            with self.subTest(model=name):
                config = configs[0]
                model = build_model_from_cfg(config)
                self.assertIsInstance(model, MiniMaxChatCompletionV2)
                model.tokens = MagicMock()
                prepared, payload = self._generate(
                    model, max_out_len=config['max_out_len'])
                self.assertEqual(prepared.url,
                                 'https://api.minimax.io/v1/chat/completions')
                self.assertEqual(model.max_seq_len, 1000000)
                self.assertEqual(payload['model'], name)
                self.assertEqual(payload['max_tokens'], output_limit)
                self.assertEqual(set(payload),
                                 {'model', 'messages', 'max_tokens'})

    @patch.dict(os.environ, {}, clear=True)
    def test_saved_legacy_and_custom_settings_are_preserved(self):
        for name, url in [
            ('MiniMax-M2.7',
             'https://api.minimax.io/v1/text/chatcompletion_v2'),
            ('MiniMax-M3', 'https://api.minimaxi.com/v1/chat/completions'),
            ('saved-custom-model',
             'https://gateway.example/v1/chat/completions'),
        ]:
            with self.subTest(model=name):
                model = self._build_model(path=name, key='saved-key', url=url,
                                          max_seq_len=204800)
                prepared, payload = self._generate(model, max_out_len=1234)
                self.assertEqual(prepared.url, url)
                self.assertEqual(prepared.headers['Authorization'],
                                 'Bearer saved-key')
                self.assertEqual(model.max_seq_len, 204800)
                self.assertEqual(payload['model'], name)
                self.assertEqual(payload['max_tokens'], 1234)

    def test_prompt_history_can_be_reused(self):
        model = self._build_model(key='test-key')
        history = PromptList([
            {'role': 'HUMAN', 'prompt': 'question'},
            {'role': 'BOT', 'prompt': '<think>reasoning</think>answer'},
            {'role': 'HUMAN', 'prompt': 'follow-up'},
        ])
        original = copy.deepcopy(history)
        _, first = self._generate(model, prompt=history)
        _, second = self._generate(model, prompt=history)
        self.assertEqual(history, original)
        self.assertEqual(first, second)
        self.assertEqual(first['messages'], [
            {'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': '<think>reasoning</think>answer'},
            {'role': 'user', 'content': 'follow-up'},
        ])

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_environment_key_fails_before_request(self):
        with patch('requests.sessions.Session.send') as send:
            with self.assertRaises(KeyError):
                self._build_model()
        send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
