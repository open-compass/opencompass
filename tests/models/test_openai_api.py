"""Unit tests for OpenAI and OpenAISDK."""

import sys
import unittest
from unittest.mock import MagicMock, patch

from opencompass.models.openai_api import OpenAI, OpenAISDK


def setup_tiktoken_mock(mock_tiktoken):
    """Helper function to setup tiktoken mock properly."""
    encode_result = [1, 2, 3, 4, 5]
    mock_enc = MagicMock()
    mock_enc.encode = MagicMock(return_value=encode_result)
    mock_tiktoken.encoding_for_model = MagicMock(return_value=mock_enc)
    mock_tiktoken.__spec__ = MagicMock()
    sys.modules['tiktoken'] = mock_tiktoken
    return mock_enc


class TestOpenAI(unittest.TestCase):
    """Test cases for OpenAI."""

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_initialization_basic(self, mock_tiktoken):
        """Test basic initialization."""
        setup_tiktoken_mock(mock_tiktoken)

        model = OpenAI(
            path='gpt-3.5-turbo',
            max_seq_len=16384,
            query_per_second=1,
        )

        self.assertEqual(model.path, 'gpt-3.5-turbo')
        self.assertEqual(model.max_seq_len, 16384)
        self.assertEqual(model.query_per_second, 1)
        self.assertEqual(model.mode, 'none')

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_initialization_with_key_list(self, mock_tiktoken):
        """Test initialization with key list."""
        setup_tiktoken_mock(mock_tiktoken)

        model = OpenAI(
            path='gpt-3.5-turbo',
            key=['key1', 'key2'],
        )

        self.assertEqual(len(model.keys), 2)
        self.assertEqual(model.keys[0], 'key1')
        self.assertEqual(model.keys[1], 'key2')

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('opencompass.models.openai_api.requests')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_basic(self, mock_requests, mock_tiktoken):
        """Test basic generate functionality."""
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]
        mock_tiktoken.encoding_for_model.return_value = mock_enc

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

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_get_token_len(self, mock_tiktoken):
        """Test get_token_len method."""
        setup_tiktoken_mock(mock_tiktoken)

        model = OpenAI(
            path='gpt-3.5-turbo',
            max_seq_len=16384,
        )

        token_len = model.get_token_len('Hello')

        self.assertEqual(token_len, 5)

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('opencompass.models.openai_api.requests')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_with_reasoning_content(self, mock_requests,
                                             mock_tiktoken):
        """Test generate with reasoning content."""
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]
        mock_tiktoken.encoding_for_model.return_value = mock_enc

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

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('opencompass.models.openai_api.requests')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_with_retry(self, mock_requests, mock_tiktoken):
        """Test generate with retry on failure."""
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]
        mock_tiktoken.encoding_for_model.return_value = mock_enc

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

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_initialization_basic(self, mock_httpx_client, mock_openai_class,
                                  mock_tiktoken):
        """Test basic initialization."""
        mock_enc = MagicMock()
        mock_tiktoken.encoding_for_model.return_value = mock_enc
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

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_basic(self, mock_httpx_client, mock_openai_class,
                            mock_tiktoken):
        """Test basic generate functionality."""
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]
        mock_tiktoken.encoding_for_model.return_value = mock_enc

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

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_with_reasoning_content(self, mock_httpx_client,
                                             mock_openai_class, mock_tiktoken):
        """Test generate with reasoning content."""
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]
        mock_tiktoken.encoding_for_model.return_value = mock_enc

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

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_with_o1_model(self, mock_httpx_client, mock_openai_class,
                                    mock_tiktoken):
        """Test generate with O1 model (reasoning model)."""
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]
        mock_tiktoken.encoding_for_model.return_value = mock_enc

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

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_with_retry(self, mock_httpx_client, mock_openai_class,
                                 mock_tiktoken):
        """Test generate with retry on failure."""
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]
        mock_tiktoken.encoding_for_model.return_value = mock_enc

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


if __name__ == '__main__':
    unittest.main()
