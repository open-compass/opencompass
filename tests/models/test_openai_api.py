"""Unit tests for OpenAI and OpenAISDK."""

import unittest
from unittest.mock import MagicMock, patch

from opencompass.models.openai_api import (OpenAI, OpenAISDK,
                                           OpenAISDKCompletion)


def setup_tiktoken_mock(mock_encoding):
    """Mock the encoder factory without replacing the tokenizer module."""
    mock_enc = MagicMock()
    mock_enc.encode.return_value = [1, 2, 3, 4, 5]
    mock_encoding.return_value = mock_enc
    return mock_enc


class TestOpenAI(unittest.TestCase):
    """Test cases for OpenAI."""

    @patch('tiktoken.encoding_for_model')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_initialization_basic(self, mock_encoding):
        """Test basic initialization."""
        setup_tiktoken_mock(mock_encoding)

        model = OpenAI(
            path='gpt-3.5-turbo',
            max_seq_len=16384,
            query_per_second=1,
        )

        self.assertEqual(model.path, 'gpt-3.5-turbo')
        self.assertEqual(model.max_seq_len, 16384)
        self.assertEqual(model.query_per_second, 1)
        self.assertEqual(model.mode, 'none')

    @patch('tiktoken.encoding_for_model')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_initialization_with_key_list(self, mock_encoding):
        """Test initialization with key list."""
        setup_tiktoken_mock(mock_encoding)

        model = OpenAI(
            path='gpt-3.5-turbo',
            key=['key1', 'key2'],
        )

        self.assertEqual(len(model.keys), 2)
        self.assertEqual(model.keys[0], 'key1')
        self.assertEqual(model.keys[1], 'key2')

    @patch('tiktoken.encoding_for_model')
    @patch('opencompass.models.openai_api.requests')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_basic(self, mock_requests, mock_encoding):
        """Test basic generate functionality."""
        setup_tiktoken_mock(mock_encoding)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Generated response'
                }
            }]
        }
        mock_requests.post.return_value = mock_response

        model = OpenAI(
            path='gpt-3.5-turbo',
            max_seq_len=16384,
        )

        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], 'Generated response')
        mock_requests.post.assert_called()

    @patch('tiktoken.encoding_for_model')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_get_token_len(self, mock_encoding):
        """Test get_token_len method."""
        setup_tiktoken_mock(mock_encoding)

        model = OpenAI(
            path='gpt-3.5-turbo',
            max_seq_len=16384,
        )

        token_len = model.get_token_len('Hello')

        self.assertEqual(token_len, 5)

    @patch('tiktoken.encoding_for_model')
    @patch('opencompass.models.openai_api.requests')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_with_reasoning_content(self, mock_requests,
                                             mock_encoding):
        """Test generate with reasoning content."""
        setup_tiktoken_mock(mock_encoding)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Final answer',
                    'reasoning_content': 'Thinking process'
                }
            }]
        }
        mock_requests.post.return_value = mock_response

        model = OpenAI(
            path='gpt-3.5-turbo',
            max_seq_len=16384,
            think_tag='</think>',
        )

        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], 'Thinking process</think>Final answer')

    @patch('tiktoken.encoding_for_model')
    @patch('opencompass.models.openai_api.requests')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_with_retry(self, mock_requests, mock_encoding):
        """Test generate with retry on failure."""
        setup_tiktoken_mock(mock_encoding)

        # First call fails, second succeeds
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Generated response'
                }
            }]
        }
        mock_requests.post.side_effect = [
            mock_response_fail, mock_response_success
        ]

        model = OpenAI(
            path='gpt-3.5-turbo',
            max_seq_len=16384,
            retry=2,
        )

        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], 'Generated response')
        self.assertEqual(mock_requests.post.call_count, 2)


class TestOpenAISDK(unittest.TestCase):
    """Test cases for OpenAISDK."""

    @patch('tiktoken.encoding_for_model')
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_initialization_basic(self, mock_httpx_client, mock_openai_class,
                                  mock_encoding):
        """Test basic initialization."""
        mock_enc = MagicMock()
        mock_encoding.return_value = mock_enc
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_httpx_client.return_value = MagicMock()

        model = OpenAISDK(
            path='gpt-3.5-turbo',
            max_seq_len=16384,
        )

        self.assertEqual(model.path, 'gpt-3.5-turbo')
        self.assertEqual(model.max_seq_len, 16384)
        self.assertEqual(model.openai_client, mock_client)

    @patch('opencompass.models.openai_api.inspect.signature')
    @patch('tiktoken.encoding_for_model')
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_proxy_config_is_passed_to_http_client(self, mock_httpx_client,
                                                   mock_openai_class,
                                                   mock_encoding,
                                                   mock_signature):
        """Test OpenAISDK proxy configuration with modern httpx."""
        mock_enc = MagicMock()
        mock_encoding.return_value = mock_enc
        mock_openai_class.return_value = MagicMock()
        mock_httpx_client.return_value = MagicMock()
        mock_signature.return_value = type('Signature', (),
                                           {'parameters': {
                                               'proxy': object()
                                           }})()

        OpenAISDK(
            path='gpt-3.5-turbo',
            openai_proxy_url='http://proxy.example',
            http_client_cfg={
                'proxy': 'http://old-proxy.example',
                'proxies': {
                    'http://': 'http://old-proxy.example'
                },
            },
        )

        http_client_kwargs = mock_httpx_client.call_args[1]
        self.assertEqual(http_client_kwargs['proxy'], 'http://proxy.example')
        self.assertNotIn('proxies', http_client_kwargs)

    @patch('tiktoken.encoding_for_model')
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_basic(self, mock_httpx_client, mock_openai_class,
                            mock_encoding):
        """Test basic generate functionality."""
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]
        mock_encoding.return_value = mock_enc

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Generated response'
        mock_response.choices[0].message.reasoning_content = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        mock_httpx_client.return_value = MagicMock()

        model = OpenAISDK(
            path='gpt-3.5-turbo',
            max_seq_len=16384,
        )

        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], 'Generated response')
        mock_client.chat.completions.create.assert_called()

    @patch('tiktoken.encoding_for_model')
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_multimodal_messages(self, mock_httpx_client,
                                          mock_openai_class, mock_encoding):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]
        mock_encoding.return_value = mock_enc

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'A'
        mock_response.choices[0].message.reasoning_content = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        mock_httpx_client.return_value = MagicMock()

        model = OpenAISDK(path='gpt-4o', max_seq_len=16384)
        model.generate([[
            dict(role='user',
                 content=[
                     dict(type='text', text='Question'),
                     dict(type='image', image_url='data:image/png;base64,AAAA')
                 ])
        ]],
                       max_out_len=100)

        messages = mock_client.chat.completions.create.call_args[1]['messages']
        self.assertEqual(messages[0]['content'][0],
                         dict(type='text', text='Question'))
        self.assertEqual(
            messages[0]['content'][1], {
                'type': 'image_url',
                'image_url': {
                    'url': 'data:image/png;base64,AAAA'
                }
            })

    @patch('tiktoken.encoding_for_model')
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_with_reasoning_content(self, mock_httpx_client,
                                             mock_openai_class, mock_encoding):
        """Test generate with reasoning content."""
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]
        mock_encoding.return_value = mock_enc

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Final answer'
        mock_response.choices[0].message.reasoning_content = 'Thinking process'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        mock_httpx_client.return_value = MagicMock()

        model = OpenAISDK(
            path='gpt-3.5-turbo',
            max_seq_len=16384,
            think_tag='</think>',
        )

        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], 'Thinking process</think>Final answer')

    @patch('tiktoken.encoding_for_model')
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_with_o1_model(self, mock_httpx_client, mock_openai_class,
                                    mock_encoding):
        """Test generate with O1 model (reasoning model)."""
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]
        mock_encoding.return_value = mock_enc

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Generated response'
        mock_response.choices[0].message.reasoning_content = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        mock_httpx_client.return_value = MagicMock()

        model = OpenAISDK(
            path='o1',
            max_seq_len=16384,
        )

        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(len(results), 1)
        # Verify max_completion_tokens is used instead of max_tokens
        call_args = mock_client.chat.completions.create.call_args[1]
        self.assertIn('max_completion_tokens', call_args)
        self.assertNotIn('max_tokens', call_args)

    @patch('tiktoken.encoding_for_model')
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_with_retry(self, mock_httpx_client, mock_openai_class,
                                 mock_encoding):
        """Test generate with retry on failure."""
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]
        mock_encoding.return_value = mock_enc

        mock_client = MagicMock()
        # First call fails, second succeeds
        from openai import APIStatusError
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Generated response'
        mock_response.choices[0].message.reasoning_content = None

        # Create APIStatusError with correct parameters
        # APIStatusError typically needs status_code and response
        mock_error_response = MagicMock()
        mock_error_response.status_code = 500
        error = APIStatusError(message='Error',
                               response=mock_error_response,
                               body=None)
        error.status_code = 500

        mock_client.chat.completions.create.side_effect = [
            error, mock_response
        ]
        mock_openai_class.return_value = mock_client
        mock_httpx_client.return_value = MagicMock()

        model = OpenAISDK(
            path='gpt-3.5-turbo',
            max_seq_len=16384,
            retry=2,
        )

        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], 'Generated response')
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)


class TestOpenAISDKCompletion(unittest.TestCase):
    """Test cases for OpenAISDKCompletion."""

    @patch('tiktoken.encoding_for_model')
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_uses_completion_api(self, mock_httpx_client,
                                          mock_openai_class, mock_encoding):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]
        mock_encoding.return_value = mock_enc

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(text='Generated response')]
        mock_client.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        mock_httpx_client.return_value = MagicMock()

        model = OpenAISDKCompletion(path='gpt-3.5-turbo')

        results = model.generate(['Hello'], max_out_len=100, temperature=0.2)

        self.assertEqual(results, ['Generated response'])
        mock_client.completions.create.assert_called_once_with(
            model='gpt-3.5-turbo',
            prompt='Hello',
            max_tokens=100,
            n=1,
            temperature=0.2,
            extra_body=None,
            timeout=3600,
        )
        mock_client.chat.completions.create.assert_not_called()

    @patch('tiktoken.encoding_for_model')
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_prompt_list_is_converted_to_text(self, mock_httpx_client,
                                              mock_openai_class,
                                              mock_encoding):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]
        mock_encoding.return_value = mock_enc

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(text='A')]
        mock_client.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        mock_httpx_client.return_value = MagicMock()

        model = OpenAISDKCompletion(path='gpt-3.5-turbo')
        model.generate([[{
            'role': 'HUMAN',
            'prompt': 'Question'
        }, {
            'role': 'BOT',
            'prompt': 'Answer:'
        }]],
                       max_out_len=10)

        self.assertEqual(
            mock_client.completions.create.call_args.kwargs['prompt'],
            'Question\nAnswer:')

    @patch('tiktoken.encoding_for_model')
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_retries_completion_request(self, mock_httpx_client,
                                                 mock_openai_class,
                                                 mock_encoding):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]
        mock_encoding.return_value = mock_enc

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(text='Generated response')]
        mock_client.completions.create.side_effect = [
            RuntimeError('temporary failure'), mock_response
        ]
        mock_openai_class.return_value = mock_client
        mock_httpx_client.return_value = MagicMock()

        model = OpenAISDKCompletion(path='gpt-3.5-turbo', retry=2)

        self.assertEqual(model.generate(['Hello'], max_out_len=100),
                         ['Generated response'])
        self.assertEqual(mock_client.completions.create.call_count, 2)

    @patch('tiktoken.encoding_for_model')
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_non_text_prompts_are_rejected_before_tokenization(
            self, mock_httpx_client, mock_openai_class, mock_encoding):
        setup_tiktoken_mock(mock_encoding)
        model = OpenAISDKCompletion(path='gpt-3.5-turbo')
        model.get_token_len = MagicMock(
            side_effect=lambda text: len(text.encode('utf-8')))

        for mode in ('none', 'front'):
            model.mode = mode
            for message in [
                {
                    'role': 'HUMAN',
                    'prompt': [{
                        'type': 'image'
                    }]
                },
                {
                    'role': 'user',
                    'content': [{
                        'type': 'image_url'
                    }]
                },
            ]:
                with self.subTest(mode=mode, message=message):
                    with self.assertRaisesRegex(TypeError,
                                                'only supports text prompts'):
                        model.generate([[message]], max_out_len=10)

        model.openai_client.completions.create.assert_not_called()


if __name__ == '__main__':
    unittest.main()
