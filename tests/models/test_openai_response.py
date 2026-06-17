"""Unit tests for OpenAISDKResponse."""

import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from opencompass.models.openai_response import OpenAISDKResponse


class FakeAPIStatusError(Exception):

    def __init__(self, message='Error', status_code=None):
        super().__init__(message)
        self.status_code = status_code


class FakeBadRequestError(FakeAPIStatusError):
    pass


@contextmanager
def fake_openai_module(client):
    module = types.ModuleType('openai')
    module.OpenAI = MagicMock(return_value=client)
    module.APIStatusError = FakeAPIStatusError
    module.BadRequestError = FakeBadRequestError
    with patch.dict(sys.modules, {'openai': module}):
        yield module


def setup_tiktoken_mock(mock_tiktoken):
    mock_enc = MagicMock()
    mock_enc.encode = MagicMock(return_value=[1, 2, 3])
    mock_tiktoken.encoding_for_model = MagicMock(return_value=mock_enc)
    mock_tiktoken.model.MODEL_TO_ENCODING = {'gpt-4.1': 'o200k_base'}
    mock_tiktoken.__spec__ = MagicMock()
    sys.modules['tiktoken'] = mock_tiktoken
    return mock_enc


class TestOpenAISDKResponse(unittest.TestCase):

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_basic(self, mock_httpx_client, mock_tiktoken):
        setup_tiktoken_mock(mock_tiktoken)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = 'Generated response'
        mock_client.responses.create.return_value = mock_response
        mock_httpx_client.return_value = MagicMock()

        with fake_openai_module(mock_client):
            model = OpenAISDKResponse(path='gpt-4.1', max_seq_len=16384)
            results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(results, ['Generated response'])
        call_args = mock_client.responses.create.call_args[1]
        self.assertEqual(call_args['model'], 'gpt-4.1')
        self.assertEqual(call_args['max_output_tokens'], 100)
        self.assertEqual(call_args['input'], [{
            'role': 'user',
            'content': 'Hello'
        }])

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_with_messages_and_extra_kwargs(
            self, mock_httpx_client, mock_tiktoken):
        setup_tiktoken_mock(mock_tiktoken)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = 'Message response'
        mock_client.responses.create.return_value = mock_response
        mock_httpx_client.return_value = MagicMock()

        messages = [{
            'role': 'developer',
            'content': 'You are concise.',
            'type': 'message',
        }, {
            'role': 'system',
            'content': 'Use plain text.'
        }, {
            'role': 'user',
            'content': 'Hello'
        }]

        with fake_openai_module(mock_client):
            model = OpenAISDKResponse(
                path='gpt-4.1',
                openai_extra_kwargs={
                    'metadata': {
                        'suite': 'unit'
                    },
                    'include': ['reasoning.encrypted_content'],
                },
                response_kwargs={
                    'tools': [{
                        'type': 'web_search_preview'
                    }],
                    'truncation': 'auto',
                },
            )
            results = model.generate([messages], max_out_len=64)

        self.assertEqual(results, ['Message response'])
        call_args = mock_client.responses.create.call_args[1]
        self.assertEqual(call_args['input'], messages)
        self.assertEqual(call_args['tools'], [{'type': 'web_search_preview'}])
        self.assertEqual(call_args['truncation'], 'auto')
        self.assertEqual(call_args['metadata'], {'suite': 'unit'})
        self.assertEqual(call_args['include'],
                         ['reasoning.encrypted_content'])

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_with_logprobs(self, mock_httpx_client, mock_tiktoken):
        setup_tiktoken_mock(mock_tiktoken)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = 'Generated response'
        mock_client.responses.create.return_value = mock_response
        mock_httpx_client.return_value = MagicMock()

        with fake_openai_module(mock_client):
            model = OpenAISDKResponse(
                path='gpt-4.1',
                logprobs=True,
                top_logprobs=3,
                openai_extra_kwargs={'include': ['reasoning.encrypted_content']},
            )
            results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(results, ['Generated response'])
        call_args = mock_client.responses.create.call_args[1]
        self.assertEqual(call_args['top_logprobs'], 3)
        self.assertEqual(call_args['include'], [
            'reasoning.encrypted_content', 'message.output_text.logprobs'
        ])

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_extracts_output_items(self, mock_httpx_client,
                                            mock_tiktoken):
        setup_tiktoken_mock(mock_tiktoken)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = None
        mock_response.output = [{
            'type': 'message',
            'content': [{
                'type': 'output_text',
                'text': 'Generated from output'
            }]
        }]
        mock_client.responses.create.return_value = mock_response
        mock_httpx_client.return_value = MagicMock()

        with fake_openai_module(mock_client):
            model = OpenAISDKResponse(path='gpt-4.1')
            results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(results, ['Generated from output'])

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_can_include_reasoning_content(self, mock_httpx_client,
                                                    mock_tiktoken):
        setup_tiktoken_mock(mock_tiktoken)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = 'Final answer'
        mock_response.output = [{
            'type': 'reasoning',
            'summary': [{
                'text': 'Thinking process'
            }]
        }, {
            'type': 'message',
            'content': [{
                'type': 'output_text',
                'text': 'Final answer'
            }]
        }]
        mock_client.responses.create.return_value = mock_response
        mock_httpx_client.return_value = MagicMock()

        with fake_openai_module(mock_client):
            model = OpenAISDKResponse(
                path='gpt-4.1',
                include_reasoning_content=True,
                think_tag='</think>',
            )
            results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(results, ['Thinking process</think>Final answer'])

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_with_status_code_mapping(self, mock_httpx_client,
                                               mock_tiktoken):
        setup_tiktoken_mock(mock_tiktoken)

        mock_client = MagicMock()
        mock_client.responses.create.side_effect = FakeBadRequestError(
            status_code=400)
        mock_httpx_client.return_value = MagicMock()

        with fake_openai_module(mock_client):
            model = OpenAISDKResponse(
                path='gpt-4.1',
                status_code_mappings={400: 'blocked'},
            )
            results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(results, ['blocked'])

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_retries_empty_incomplete_response(
            self, mock_httpx_client, mock_tiktoken):
        setup_tiktoken_mock(mock_tiktoken)

        mock_client = MagicMock()
        empty_response = MagicMock()
        empty_response.output_text = None
        empty_response.output = []
        empty_response.incomplete_details = {'reason': 'max_output_tokens'}
        empty_response.error = None
        empty_response.status = 'incomplete'
        success_response = MagicMock()
        success_response.output_text = 'After retry'
        mock_client.responses.create.side_effect = [
            empty_response, success_response
        ]
        mock_httpx_client.return_value = MagicMock()

        with fake_openai_module(mock_client):
            model = OpenAISDKResponse(path='gpt-4.1', retry=2)
            results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(results, ['After retry'])
        self.assertEqual(mock_client.responses.create.call_count, 2)

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_proxy_config_is_passed_to_http_client(self, mock_httpx_client,
                                                  mock_tiktoken):
        setup_tiktoken_mock(mock_tiktoken)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = 'Generated response'
        mock_client.responses.create.return_value = mock_response

        with fake_openai_module(mock_client):
            OpenAISDKResponse(path='gpt-4.1',
                              openai_proxy_url='http://proxy.example')

        http_client_kwargs = mock_httpx_client.call_args[1]
        self.assertTrue('proxy' in http_client_kwargs
                        or 'proxies' in http_client_kwargs)


if __name__ == '__main__':
    unittest.main()
