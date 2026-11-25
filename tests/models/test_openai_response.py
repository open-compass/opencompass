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
    def test_reasoning_model_omits_temperature(self, mock_httpx_client,
                                               mock_tiktoken):
        setup_tiktoken_mock(mock_tiktoken)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = 'Generated response'
        mock_client.responses.create.return_value = mock_response
        mock_httpx_client.return_value = MagicMock()

        with fake_openai_module(mock_client):
            model = OpenAISDKResponse(path='gpt-5.1', max_seq_len=16384)
            results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(results, ['Generated response'])
        call_args = mock_client.responses.create.call_args[1]
        self.assertNotIn('temperature', call_args)

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_non_reasoning_model_passes_temperature(self, mock_httpx_client,
                                                    mock_tiktoken):
        setup_tiktoken_mock(mock_tiktoken)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = 'Generated response'
        mock_client.responses.create.return_value = mock_response
        mock_httpx_client.return_value = MagicMock()

        with fake_openai_module(mock_client):
            model = OpenAISDKResponse(path='gpt-4.1',
                                      max_seq_len=16384,
                                      temperature=0.7)
            results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(results, ['Generated response'])
        call_args = mock_client.responses.create.call_args[1]
        self.assertEqual(call_args['temperature'], 0.7)

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
    def test_generate_includes_reasoning_content_when_returned(
            self, mock_httpx_client, mock_tiktoken):
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
                think_tag='</think>',
            )
            results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(results, ['Thinking process</think>Final answer'])

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_reasoning_content_takes_priority_over_summary(
            self, mock_httpx_client, mock_tiktoken):
        setup_tiktoken_mock(mock_tiktoken)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = 'Final answer'
        mock_response.output = [{
            'type': 'reasoning',
            'summary': [{
                'text': 'Thinking summary'
            }],
            'content': [{
                'text': 'Raw reasoning'
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
                think_tag='</think>',
            )
            results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(results, ['Raw reasoning</think>Final answer'])

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_reasoning_model_requests_summary(
            self, mock_httpx_client, mock_tiktoken):
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
        extra_kwargs = {'reasoning': {'effort': 'high'}}

        with fake_openai_module(mock_client):
            model = OpenAISDKResponse(
                path='gpt-5.1',
                openai_extra_kwargs=extra_kwargs,
            )
            results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(results, ['Thinking process</think>Final answer'])
        call_args = mock_client.responses.create.call_args[1]
        self.assertEqual(call_args['reasoning'], {
            'effort': 'high',
            'summary': 'auto',
        })
        self.assertNotIn('temperature', call_args)
        self.assertEqual(extra_kwargs, {'reasoning': {'effort': 'high'}})

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


    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_streaming_collects_text_deltas(self, mock_httpx_client,
                                                     mock_tiktoken):
        setup_tiktoken_mock(mock_tiktoken)

        mock_client = MagicMock()
        mock_client.responses.create.return_value = [
            {
                'type': 'response.output_text.delta',
                'delta': 'Hello'
            },
            {
                'type': 'response.output_text.delta',
                'delta': ' world'
            },
            {
                'type': 'response.completed',
                'response': {
                    'status': 'completed',
                    'output': []
                }
            },
        ]
        mock_httpx_client.return_value = MagicMock()

        with fake_openai_module(mock_client):
            model = OpenAISDKResponse(path='gpt-4.1', stream=True)
            results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(results, ['Hello world'])
        call_args = mock_client.responses.create.call_args[1]
        self.assertTrue(call_args['stream'])

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_streaming_includes_reasoning_content_when_returned(
            self, mock_httpx_client, mock_tiktoken):
        setup_tiktoken_mock(mock_tiktoken)

        mock_client = MagicMock()
        mock_client.responses.create.return_value = [
            {
                'type': 'response.reasoning_summary_text.delta',
                'delta': 'Thinking process'
            },
            {
                'type': 'response.output_text.delta',
                'delta': 'Final answer'
            },
            {
                'type': 'response.completed',
                'response': {
                    'status': 'completed',
                    'output': []
                }
            },
        ]
        mock_httpx_client.return_value = MagicMock()

        with fake_openai_module(mock_client):
            model = OpenAISDKResponse(
                path='gpt-4.1',
                stream=True,
                think_tag='</think>',
            )
            results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(results, ['Thinking process</think>Final answer'])
        call_args = mock_client.responses.create.call_args[1]
        self.assertTrue(call_args['stream'])

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_streaming_reasoning_text_takes_priority_over_summary(
            self, mock_httpx_client, mock_tiktoken):
        setup_tiktoken_mock(mock_tiktoken)

        mock_client = MagicMock()
        mock_client.responses.create.return_value = [
            {
                'type': 'response.reasoning_summary_text.delta',
                'delta': 'Thinking summary'
            },
            {
                'type': 'response.reasoning_text.delta',
                'delta': 'Raw reasoning'
            },
            {
                'type': 'response.output_text.delta',
                'delta': 'Final answer'
            },
            {
                'type': 'response.completed',
                'response': {
                    'status': 'completed',
                    'output': []
                }
            },
        ]
        mock_httpx_client.return_value = MagicMock()

        with fake_openai_module(mock_client):
            model = OpenAISDKResponse(
                path='gpt-4.1',
                stream=True,
                think_tag='</think>',
            )
            results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(results, ['Raw reasoning</think>Final answer'])

    PARTIAL = [
        {
            'type': 'response.reasoning_summary_text.delta',
            'delta': 'Thinking'
        },
        {
            'type': 'response.output_text.delta',
            'delta': 'par'
        },
    ]
    FULL = [
        {
            'type': 'response.output_text.delta',
            'delta': 'full'
        },
        {
            'type': 'response.completed',
            'response': {
                'status': 'completed',
                'output': []
            }
        },
    ]

    def run_responses(self, responses, **kwargs):
        """Generate once, returning the model, client and a callable."""
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = responses
        kwargs = {'stream': True, 'retry': 2, **kwargs}
        with patch('opencompass.models.openai_api.tiktoken',
                   create=True) as mock_tiktoken, \
                patch('httpx.Client'), \
                patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}), \
                fake_openai_module(mock_client):
            setup_tiktoken_mock(mock_tiktoken)
            model = OpenAISDKResponse(path='gpt-4.1', **kwargs)
            model.logger = MagicMock()
            try:
                result = model.generate(['Hello'], max_out_len=100)[0]
            except RuntimeError as e:
                result = e
        return result, mock_client.responses.create.call_count, model

    def test_truncated_stream_is_retried(self):
        """Test streams cut in thinking or in text are requested again."""
        for cut in (self.PARTIAL[:1], self.PARTIAL):
            with self.subTest(events=len(cut)):
                result, calls, _ = self.run_responses([cut, self.FULL])
                self.assertEqual(result, 'full')
                self.assertEqual(calls, 2)

    def test_truncated_stream_fails_after_retries(self):
        """Test a truncated reply is never returned as the prediction."""
        result, calls, _ = self.run_responses([self.PARTIAL, self.PARTIAL])
        self.assertIsInstance(result, RuntimeError)
        self.assertEqual(calls, 2)

    def test_failed_stream_with_partial_text_is_retried(self):
        """Test failed and error events discard the partial text."""
        ends = [
            {
                'type': 'response.failed',
                'response': {
                    'status': 'failed',
                    'error': {
                        'code': 'server_error'
                    }
                }
            },
            {
                'type': 'error',
                'code': 'server_error',
                'message': 'boom'
            },
        ]
        for end in ends:
            for confirm in (True, False):
                with self.subTest(event=end['type'], confirm=confirm):
                    result, calls, _ = self.run_responses(
                        [self.PARTIAL + [end], self.FULL],
                        finish_reason_confirm=confirm)
                    self.assertEqual(result, 'full')
                    self.assertEqual(calls, 2)

    def test_failed_response_with_partial_text_is_retried(self):
        """Test non-streaming failed responses discard the partial text."""
        result, calls, _ = self.run_responses([
            types.SimpleNamespace(
                status='failed', output_text='par', error='boom'),
            types.SimpleNamespace(status='completed', output_text='full'),
        ], stream=False)
        self.assertEqual(result, 'full')
        self.assertEqual(calls, 2)

    def test_incomplete_stream_is_not_retried(self):
        """Test incomplete is a terminal state, warning on max tokens."""
        for reason, warned in (('max_output_tokens', True),
                               ('content_filter', False)):
            with self.subTest(reason=reason):
                end = {
                    'type': 'response.incomplete',
                    'response': {
                        'status': 'incomplete',
                        'incomplete_details': {
                            'reason': reason
                        }
                    }
                }
                result, calls, model = self.run_responses(
                    [self.PARTIAL + [end]])
                self.assertEqual(result, 'Thinking</think>par')
                self.assertEqual(calls, 1)
                messages = [
                    str(c) for c in model.logger.warning.call_args_list
                ]
                self.assertEqual(
                    any('max_output_tokens' in m for m in messages), warned)

    def test_finish_reason_confirm_can_be_disabled(self):
        """Test finish_reason_confirm=False keeps the truncated reply."""
        result, calls, _ = self.run_responses([self.PARTIAL],
                                              finish_reason_confirm=False)
        self.assertEqual(result, 'Thinking</think>par')
        self.assertEqual(calls, 1)


if __name__ == '__main__':
    unittest.main()
