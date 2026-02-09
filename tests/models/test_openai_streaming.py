"""Unit tests for OpenAISDKStreaming."""

import sys
import unittest
from unittest.mock import MagicMock, patch

from opencompass.models.openai_streaming import OpenAISDKStreaming


def setup_tiktoken_mock(mock_tiktoken):
    """Helper function to setup tiktoken mock properly."""
    encode_result = [1, 2, 3, 4, 5]
    mock_enc = MagicMock()
    mock_enc.encode = MagicMock(return_value=encode_result)
    mock_tiktoken.encoding_for_model = MagicMock(return_value=mock_enc)
    mock_tiktoken.__spec__ = MagicMock()
    sys.modules['tiktoken'] = mock_tiktoken
    return mock_enc


class TestOpenAISDKStreaming(unittest.TestCase):
    """Test cases for OpenAISDKStreaming."""

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_initialization_basic(self, mock_openai_class, mock_tiktoken):
        """Test basic initialization."""
        mock_enc = MagicMock()
        mock_tiktoken.encoding_for_model.return_value = mock_enc
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        model = OpenAISDKStreaming(
            path='gpt-3.5-turbo',
            max_seq_len=16384,
            stream=True,
        )

        self.assertEqual(model.path, 'gpt-3.5-turbo')
        self.assertEqual(model.max_seq_len, 16384)
        self.assertTrue(model.stream)
        self.assertEqual(model.stream_chunk_size, 1)
        self.assertEqual(model.openai_client, mock_client)

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_initialization_with_stream_false(self, mock_openai_class,
                                              mock_tiktoken):
        """Test initialization with stream=False."""
        mock_enc = MagicMock()
        mock_tiktoken.encoding_for_model.return_value = mock_enc
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        model = OpenAISDKStreaming(
            path='gpt-3.5-turbo',
            max_seq_len=16384,
            stream=False,
        )

        self.assertFalse(model.stream)

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_with_streaming(self, mock_openai_class, mock_tiktoken):
        """Test generate with streaming enabled."""
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        # Mock streaming response
        mock_chunk1 = MagicMock()
        mock_delta1 = MagicMock()
        mock_delta1.content = 'Hello'
        mock_delta1.reasoning_content = None
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta = mock_delta1
        mock_chunk1.choices[0].finish_reason = None

        mock_chunk2 = MagicMock()
        mock_delta2 = MagicMock()
        mock_delta2.content = ' World'
        mock_delta2.reasoning_content = None
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta = mock_delta2
        mock_chunk2.choices[0].finish_reason = None

        mock_chunk3 = MagicMock()
        mock_delta3 = MagicMock()
        mock_delta3.content = None
        mock_delta3.reasoning_content = None
        mock_chunk3.choices = [MagicMock()]
        mock_chunk3.choices[0].delta = mock_delta3
        mock_chunk3.choices[0].finish_reason = 'stop'

        mock_stream = iter([mock_chunk1, mock_chunk2, mock_chunk3])

        mock_fresh_client = MagicMock()
        mock_fresh_client.chat.completions.create.return_value = mock_stream
        mock_fresh_client._client = MagicMock()
        mock_fresh_client._client.close = MagicMock()

        # Mock _create_fresh_client
        with patch.object(OpenAISDKStreaming,
                          '_create_fresh_client') as mock_create_client:
            mock_create_client.return_value = mock_fresh_client

            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            model = OpenAISDKStreaming(
                path='gpt-3.5-turbo',
                max_seq_len=16384,
                stream=True,
            )

            results = model.generate(['Hello'], max_out_len=100)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0], 'Hello World')
            mock_fresh_client.chat.completions.create.assert_called_once()

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_with_streaming_reasoning_content(self, mock_openai_class,
                                                       mock_tiktoken):
        """Test generate with streaming and reasoning content."""
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        # Mock streaming response with reasoning content
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.reasoning_content = 'Thinking'
        mock_chunk1.choices[0].delta.content = None
        mock_chunk1.choices[0].finish_reason = None

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.reasoning_content = ' process'
        mock_chunk2.choices[0].delta.content = None
        mock_chunk2.choices[0].finish_reason = None

        mock_chunk3 = MagicMock()
        mock_chunk3.choices = [MagicMock()]
        mock_chunk3.choices[0].delta.reasoning_content = None
        mock_chunk3.choices[0].delta.content = 'Answer'
        mock_chunk3.choices[0].finish_reason = None

        mock_chunk4 = MagicMock()
        mock_chunk4.choices = [MagicMock()]
        mock_chunk4.choices[0].delta.reasoning_content = None
        mock_chunk4.choices[0].delta.content = None
        mock_chunk4.choices[0].finish_reason = 'stop'

        mock_stream = iter(
            [mock_chunk1, mock_chunk2, mock_chunk3, mock_chunk4])

        mock_fresh_client = MagicMock()
        mock_fresh_client.chat.completions.create.return_value = mock_stream
        mock_fresh_client._client = MagicMock()
        mock_fresh_client._client.close = MagicMock()

        with patch.object(OpenAISDKStreaming,
                          '_create_fresh_client') as mock_create_client:
            mock_create_client.return_value = mock_fresh_client

            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            model = OpenAISDKStreaming(
                path='gpt-3.5-turbo',
                max_seq_len=16384,
                stream=True,
                think_tag='</think>',
            )

            results = model.generate(['Hello'], max_out_len=100)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0], 'Thinking process</think>Answer')

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_create_fresh_client(self, mock_openai_class, mock_tiktoken):
        """Test _create_fresh_client method."""
        mock_enc = MagicMock()
        mock_tiktoken.encoding_for_model.return_value = mock_enc
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        model = OpenAISDKStreaming(
            path='gpt-3.5-turbo',
            max_seq_len=16384,
        )

        fresh_client = model._create_fresh_client()

        self.assertIsNotNone(fresh_client)
        mock_openai_class.assert_called()

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_estimate_token_count(self, mock_openai_class, mock_tiktoken):
        """Test estimate_token_count method."""
        setup_tiktoken_mock(mock_tiktoken)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        model = OpenAISDKStreaming(
            path='gpt-3.5-turbo',
            max_seq_len=16384,
        )

        token_count = model.estimate_token_count('Hello')

        self.assertEqual(token_count, 5)


if __name__ == '__main__':
    unittest.main()
