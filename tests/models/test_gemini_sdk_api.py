"""Unit tests for GeminiSDK."""

import types
import unittest
from unittest.mock import MagicMock, patch

from opencompass.models.gemini_sdk_api import GeminiSDK
from opencompass.utils.prompt import PromptList


class TestGeminiSDK(unittest.TestCase):
    """Test cases for GeminiSDK."""

    def build_model(self, **kwargs):
        """Build GeminiSDK with a fake google.genai module."""
        key = kwargs.pop('key', 'test-key')
        client = MagicMock()
        client.models.generate_content.return_value = types.SimpleNamespace(
            text='OK')

        client_cls = MagicMock(return_value=client)
        http_options_cls = MagicMock(
            side_effect=lambda **kwargs: types.SimpleNamespace(**kwargs))
        thinking_config_cls = MagicMock(
            side_effect=lambda **kwargs: types.SimpleNamespace(**kwargs))
        config_cls = MagicMock(
            side_effect=lambda **kwargs: types.SimpleNamespace(**kwargs))

        genai_module = types.SimpleNamespace(Client=client_cls)
        types_module = types.SimpleNamespace(
            HttpOptions=http_options_cls,
            ThinkingConfig=thinking_config_cls,
            GenerateContentConfig=config_cls,
        )
        genai_module.types = types_module
        google_module = types.SimpleNamespace(genai=genai_module)

        with patch.dict(
                'sys.modules',
            {
                'google': google_module,
                'google.genai': genai_module,
                'google.genai.types': types_module,
            },
        ):
            model = GeminiSDK(
                key=key,
                query_per_second=1000000,
                retry=1,
                **kwargs,
            )
        model.wait = MagicMock()
        return model, client_cls, http_options_cls, config_cls, client

    @patch.dict('os.environ', {
        'GOOGLE_API_KEY': 'google-key',
        'GEMINI_API_KEY': 'gemini-key',
    },
                clear=True)
    def test_google_api_key_has_higher_priority(self):
        """Test key='ENV' uses GOOGLE_API_KEY before GEMINI_API_KEY."""
        _, client_cls, _, _, _ = self.build_model(key='ENV')

        client_cls.assert_called_once_with(api_key='google-key')

    @patch.dict('os.environ', {'GEMINI_API_KEY': 'gemini-key'}, clear=True)
    def test_gemini_api_key_is_used_as_fallback(self):
        """Test key='ENV' falls back to GEMINI_API_KEY."""
        _, client_cls, _, _, _ = self.build_model(key='ENV')

        client_cls.assert_called_once_with(api_key='gemini-key')

    @patch.dict('os.environ', {}, clear=True)
    def test_missing_env_key_raises(self):
        """Test key='ENV' requires a Gemini key in the environment."""
        with self.assertRaisesRegex(ValueError,
                                    'GOOGLE_API_KEY or GEMINI_API_KEY'):
            self.build_model(key='ENV')

    @patch.dict('os.environ',
                {'GEMINI_BASE_URL': 'https://proxy.example'},
                clear=True)
    def test_gemini_base_url_env_is_used_as_is(self):
        """Test GEMINI_BASE_URL is passed through as configured."""
        model, client_cls, http_options_cls, _, _ = self.build_model()

        http_options_cls.assert_called_once_with(
            base_url='https://proxy.example')
        client_kwargs = client_cls.call_args.kwargs
        self.assertEqual(client_kwargs['api_key'], 'test-key')
        self.assertEqual(client_kwargs['http_options'].base_url,
                         'https://proxy.example')
        self.assertEqual(model.base_url, 'https://proxy.example')

    @patch.dict('os.environ',
                {'GEMINI_BASE_URL': 'https://env.example'},
                clear=True)
    def test_explicit_base_url_overrides_env(self):
        """Test explicit base_url has higher priority than GEMINI_BASE_URL."""
        model, _, http_options_cls, _, _ = self.build_model(
            base_url='https://arg.example')

        http_options_cls.assert_called_once_with(
            base_url='https://arg.example')
        self.assertEqual(model.base_url, 'https://arg.example')

    @patch.dict('os.environ', {}, clear=True)
    def test_system_prompt_is_config_parameter(self):
        """Test SYSTEM PromptList entries are not sent as contents roles."""
        model, _, _, _, client = self.build_model(base_url='https://proxy')
        prompt = PromptList([
            {
                'role': 'SYSTEM',
                'prompt': 'system one',
            },
            {
                'role': 'HUMAN',
                'prompt': 'hello',
            },
            {
                'role': 'BOT',
                'prompt': 'hi',
            },
            {
                'role': 'SYSTEM',
                'prompt': 'system two',
            },
        ])

        self.assertEqual(model._generate(prompt, max_out_len=64), 'OK')
        api_params = client.models.generate_content.call_args.kwargs

        self.assertEqual(api_params['config'].system_instruction,
                         'system one\nsystem two')
        self.assertEqual(api_params['contents'], [
            {
                'role': 'user',
                'parts': [{
                    'text': 'hello',
                }],
            },
            {
                'role': 'model',
                'parts': [{
                    'text': 'hi',
                }],
            },
        ])

    @patch.dict('os.environ', {}, clear=True)
    def test_raw_prompt_roles_are_converted(self):
        """Test raw prompt system/user/assistant roles are supported."""
        model, _, _, _, client = self.build_model()
        prompt = [
            {
                'role': 'system',
                'content': 'raw system',
            },
            {
                'role': 'user',
                'content': 'raw user',
            },
            {
                'role': 'assistant',
                'content': 'raw assistant',
            },
        ]

        self.assertEqual(model._generate(prompt, max_out_len=64), 'OK')
        api_params = client.models.generate_content.call_args.kwargs

        self.assertEqual(api_params['config'].system_instruction, 'raw system')
        self.assertEqual(api_params['contents'], [
            {
                'role': 'user',
                'parts': [{
                    'text': 'raw user',
                }],
            },
            {
                'role': 'model',
                'parts': [{
                    'text': 'raw assistant',
                }],
            },
        ])

    @patch.dict('os.environ', {}, clear=True)
    def test_generation_config_is_built(self):
        """Test generation arguments are passed to GenerateContentConfig."""
        model, _, _, _, client = self.build_model(
            temperature=0.3,
            top_p=0.9,
            top_k=20,
            gemini_extra_kwargs={'stop_sequences': ['STOP']},
        )

        self.assertEqual(model._generate('hello', max_out_len=64), 'OK')
        api_params = client.models.generate_content.call_args.kwargs
        config = api_params['config']

        self.assertEqual(config.max_output_tokens, 64)
        self.assertEqual(config.temperature, 0.3)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.top_k, 20)
        self.assertEqual(config.candidate_count, 1)
        self.assertEqual(config.stop_sequences, ['STOP'])

    @patch.dict('os.environ', {}, clear=True)
    def test_thinking_config_is_built(self):
        """Test Gemini thinking config is passed to GenerateContentConfig."""
        model, _, _, config_cls, client = self.build_model(
            thinking={
                'include_thoughts': True,
                'thinking_budget': 1024,
            })

        self.assertEqual(model._generate('hello', max_out_len=64), 'OK')
        api_params = client.models.generate_content.call_args.kwargs
        config = api_params['config']

        self.assertTrue(config.thinking_config.include_thoughts)
        self.assertEqual(config.thinking_config.thinking_budget, 1024)
        config_kwargs = config_cls.call_args.kwargs
        self.assertIn('thinking_config', config_kwargs)

    @patch.dict('os.environ', {}, clear=True)
    def test_non_stream_response_falls_back_to_candidate_parts(self):
        """Test non-stream response parsing supports candidate parts."""
        model, _, _, _, client = self.build_model()
        client.models.generate_content.return_value = types.SimpleNamespace(
            text=None,
            candidates=[
                types.SimpleNamespace(
                    content=types.SimpleNamespace(parts=[
                        types.SimpleNamespace(text='Final answer')
                    ]))
            ],
        )

        self.assertEqual(model._generate('hello', max_out_len=64),
                         'Final answer')

    @patch.dict('os.environ', {}, clear=True)
    def test_non_stream_response_preserves_thought_parts(self):
        """Test non-stream response parsing preserves thought summaries."""
        model, _, _, _, client = self.build_model(
            thinking={'include_thoughts': True})
        client.models.generate_content.return_value = types.SimpleNamespace(
            text=None,
            candidates=[
                types.SimpleNamespace(
                    content=types.SimpleNamespace(parts=[
                        types.SimpleNamespace(
                            text='Thinking process',
                            thought=True,
                        ),
                        types.SimpleNamespace(
                            text='Final answer',
                            thought=False,
                        ),
                    ]))
            ],
        )

        self.assertEqual(model._generate('hello', max_out_len=64),
                         'Thinking process</think>Final answer')

    @patch.dict('os.environ', {}, clear=True)
    def test_stream_true_uses_streaming_response(self):
        """Test stream=True accumulates streamed text deltas."""
        model, _, _, _, client = self.build_model(stream=True)
        client.models.generate_content_stream.return_value = iter([
            types.SimpleNamespace(text='Hello'),
            types.SimpleNamespace(text=' World'),
        ])

        self.assertEqual(model._generate('hello', max_out_len=64),
                         'Hello World')
        client.models.generate_content.assert_not_called()
        client.models.generate_content_stream.assert_called_once()

    @patch.dict('os.environ', {}, clear=True)
    def test_stream_response_preserves_thought_parts(self):
        """Test stream=True accumulates thought summaries and text deltas."""
        model, _, _, _, client = self.build_model(
            stream=True, thinking={'include_thoughts': True})
        client.models.generate_content_stream.return_value = iter([
            types.SimpleNamespace(candidates=[
                types.SimpleNamespace(
                    content=types.SimpleNamespace(parts=[
                        types.SimpleNamespace(text='Thinking',
                                              thought=True)
                    ]))
            ]),
            types.SimpleNamespace(candidates=[
                types.SimpleNamespace(
                    content=types.SimpleNamespace(parts=[
                        types.SimpleNamespace(text=' process',
                                              thought=True)
                    ]))
            ]),
            types.SimpleNamespace(candidates=[
                types.SimpleNamespace(
                    content=types.SimpleNamespace(parts=[
                        types.SimpleNamespace(text='Final',
                                              thought=False)
                    ]))
            ]),
            types.SimpleNamespace(candidates=[
                types.SimpleNamespace(
                    content=types.SimpleNamespace(parts=[
                        types.SimpleNamespace(text=' answer',
                                              thought=False)
                    ]))
            ]),
        ])

        self.assertEqual(model._generate('hello', max_out_len=64),
                         'Thinking process</think>Final answer')

    @patch.dict('os.environ', {}, clear=True)
    def test_generate_multiple_inputs_uses_progress_bar(self):
        """Test batched generation displays the same progress bar as OpenAI."""
        model, _, _, _, _ = self.build_model()

        with patch('opencompass.models.gemini_sdk_api.tqdm') as mock_tqdm:
            mock_tqdm.side_effect = lambda iterable, **kwargs: iterable

            self.assertEqual(model.generate(['hello', 'world'], 64),
                             ['OK', 'OK'])

        tqdm_kwargs = mock_tqdm.call_args.kwargs
        self.assertEqual(tqdm_kwargs['total'], 2)
        self.assertEqual(tqdm_kwargs['desc'], 'Inferencing')

    @patch.dict('os.environ', {}, clear=True)
    def test_generate_single_input_skips_progress_bar(self):
        """Test single-input generation follows OpenAI's direct path."""
        model, _, _, _, _ = self.build_model()

        with patch('opencompass.models.gemini_sdk_api.tqdm') as mock_tqdm:
            self.assertEqual(model.generate(['hello'], 64), ['OK'])

        mock_tqdm.assert_not_called()


if __name__ == '__main__':
    unittest.main()
