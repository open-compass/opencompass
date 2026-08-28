"""Unit tests for OrcaRouterAPI and OrcaRouterAPIStreaming."""

import unittest
from unittest.mock import MagicMock, patch

from opencompass.models.orcarouter_api import (ORCAROUTER_API_BASE,
                                               OrcaRouterAPI,
                                               OrcaRouterAPIStreaming)


def _setup_tiktoken(mock_tiktoken):
    """Install a tiktoken stub that resolves any requested encoding."""
    mock_enc = MagicMock()
    mock_enc.encode = MagicMock(return_value=[1, 2, 3])
    mock_tiktoken.encoding_for_model = MagicMock(return_value=mock_enc)
    mock_tiktoken.model = MagicMock()
    mock_tiktoken.model.MODEL_TO_ENCODING = {}
    return mock_enc


class TestOrcaRouterAPI(unittest.TestCase):
    """Initialization and key/base handling."""

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'ORCAROUTER_API_KEY': 'sk-orca'})
    def test_registers_in_models_registry(self, mock_httpx_client,
                                          mock_openai_class, mock_tiktoken):
        from opencompass.registry import MODELS

        _setup_tiktoken(mock_tiktoken)
        mock_openai_class.return_value = MagicMock()
        mock_httpx_client.return_value = MagicMock()
        self.assertIs(MODELS.get('OrcaRouterAPI'), OrcaRouterAPI)
        self.assertIs(MODELS.get('OrcaRouterAPIStreaming'),
                      OrcaRouterAPIStreaming)

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'ORCAROUTER_API_KEY': 'sk-orca'})
    def test_default_init_uses_orcarouter_defaults(self, mock_httpx_client,
                                                   mock_openai_class,
                                                   mock_tiktoken):
        _setup_tiktoken(mock_tiktoken)
        mock_openai_class.return_value = MagicMock()
        mock_httpx_client.return_value = MagicMock()

        model = OrcaRouterAPI()

        self.assertEqual(model.path, 'orcarouter/free')
        self.assertEqual(model.openai_api_base, ORCAROUTER_API_BASE)
        self.assertEqual(model.keys, ['sk-orca'])

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'ORCAROUTER_API_KEY': 'sk-orca'})
    def test_env_key_reads_orcarouter_var(self, mock_httpx_client,
                                          mock_openai_class, mock_tiktoken):
        _setup_tiktoken(mock_tiktoken)
        mock_openai_class.return_value = MagicMock()
        mock_httpx_client.return_value = MagicMock()

        model = OrcaRouterAPI(key='ENV')

        self.assertEqual(model.keys, ['sk-orca'])

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {}, clear=True)
    def test_env_key_raises_when_missing(self, mock_httpx_client,
                                         mock_openai_class, mock_tiktoken):
        _setup_tiktoken(mock_tiktoken)
        mock_openai_class.return_value = MagicMock()
        mock_httpx_client.return_value = MagicMock()

        with self.assertRaisesRegex(ValueError, 'OrcaRouter API key'):
            OrcaRouterAPI(key='ENV')

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {}, clear=True)
    def test_explicit_key_skips_env(self, mock_httpx_client, mock_openai_class,
                                    mock_tiktoken):
        _setup_tiktoken(mock_tiktoken)
        mock_openai_class.return_value = MagicMock()
        mock_httpx_client.return_value = MagicMock()

        model = OrcaRouterAPI(key='sk-explicit')

        self.assertEqual(model.keys, ['sk-explicit'])

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'ORCAROUTER_API_KEY': 'sk-orca'})
    def test_custom_api_base_is_respected(self, mock_httpx_client,
                                          mock_openai_class, mock_tiktoken):
        _setup_tiktoken(mock_tiktoken)
        mock_openai_class.return_value = MagicMock()
        mock_httpx_client.return_value = MagicMock()

        model = OrcaRouterAPI(openai_api_base='https://self-hosted/v1/')

        self.assertEqual(model.openai_api_base, 'https://self-hosted/v1/')


class TestOrcaRouterAPIGenerate(unittest.TestCase):
    """End-to-end generate with a mocked OpenAI client."""

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'ORCAROUTER_API_KEY': 'sk-orca'})
    def test_generate_single(self, mock_httpx_client, mock_openai_class,
                             mock_tiktoken):
        _setup_tiktoken(mock_tiktoken)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'pong'
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].finish_reason = 'stop'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        mock_httpx_client.return_value = MagicMock()

        model = OrcaRouterAPI(path='orcarouter/free')
        results = model.generate(['ping'], max_out_len=16)

        self.assertEqual(results, ['pong'])
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs['model'], 'orcarouter/free')

        # The OpenAI SDK client is constructed with the gateway base URL
        # and the resolved key.
        client_kwargs = mock_openai_class.call_args[1]
        self.assertEqual(client_kwargs['base_url'], ORCAROUTER_API_BASE)
        self.assertEqual(client_kwargs['api_key'], 'sk-orca')

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'ORCAROUTER_API_KEY': 'sk-orca'})
    def test_generate_merges_reasoning_content(self, mock_httpx_client,
                                               mock_openai_class,
                                               mock_tiktoken):
        _setup_tiktoken(mock_tiktoken)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Final'
        mock_response.choices[0].message.reasoning_content = 'Thinking'
        mock_response.choices[0].finish_reason = 'stop'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        mock_httpx_client.return_value = MagicMock()

        model = OrcaRouterAPI(think_tag='</think>')
        results = model.generate(['q'], max_out_len=16)

        self.assertEqual(results, ['Thinking</think>Final'])


class TestOrcaRouterAPIStreaming(unittest.TestCase):
    """Streaming variant parses chunks back into a full response."""

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'ORCAROUTER_API_KEY': 'sk-orca'})
    def test_stream_defaults(self, mock_httpx_client, mock_openai_class,
                             mock_tiktoken):
        _setup_tiktoken(mock_tiktoken)
        mock_openai_class.return_value = MagicMock()
        mock_httpx_client.return_value = MagicMock()

        model = OrcaRouterAPIStreaming()

        self.assertEqual(model.path, 'orcarouter/free')
        self.assertEqual(model.openai_api_base, ORCAROUTER_API_BASE)
        self.assertTrue(model.stream)
        self.assertEqual(model.stream_chunk_size, 1)

    @patch('opencompass.models.openai_api.tiktoken', create=True)
    @patch('openai.OpenAI')
    @patch('httpx.Client')
    @patch.dict('os.environ', {'ORCAROUTER_API_KEY': 'sk-orca'})
    def test_stream_generate(self, mock_httpx_client, mock_openai_class,
                             mock_tiktoken):
        _setup_tiktoken(mock_tiktoken)

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = 'Hel'
        chunk1.choices[0].delta.reasoning_content = None
        chunk1.choices[0].finish_reason = None
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = 'lo'
        chunk2.choices[0].delta.reasoning_content = None
        chunk2.choices[0].finish_reason = 'stop'

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(
            [chunk1, chunk2])
        mock_openai_class.return_value = mock_client
        mock_httpx_client.return_value = MagicMock()

        model = OrcaRouterAPIStreaming()
        results = model.generate(['hi'], max_out_len=16)

        self.assertEqual(results, ['Hello'])
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertTrue(call_kwargs['stream'])
        self.assertEqual(call_kwargs['model'], 'orcarouter/free')


if __name__ == '__main__':
    unittest.main()
