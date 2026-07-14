"""Unit tests for ClaudeSDK."""

import types
import unittest
from unittest.mock import MagicMock, patch

from opencompass.models.claude_sdk_api import ClaudeSDK
from opencompass.utils.prompt import PromptList


class TestClaudeSDK(unittest.TestCase):
    """Test cases for ClaudeSDK."""

    def build_model(self, **kwargs):
        """Build ClaudeSDK with a fake anthropic module."""
        key = kwargs.pop('key', 'test-key')
        client = MagicMock()
        content = types.SimpleNamespace(type='text', text='OK')
        client.messages.create.return_value = types.SimpleNamespace(
            content=[content])
        anthropic_cls = MagicMock(return_value=client)
        anthropic_module = types.SimpleNamespace(Anthropic=anthropic_cls)

        with patch.dict('sys.modules', {'anthropic': anthropic_module}):
            model = ClaudeSDK(
                key=key,
                query_per_second=1000000,
                retry=1,
                **kwargs,
            )
        model.wait = MagicMock()
        return model, anthropic_cls, client

    @patch.dict('os.environ',
                {'ANTHROPIC_BASE_URL': 'https://proxy.example/v1'},
                clear=True)
    def test_anthropic_base_url_env_is_used_as_is(self):
        """Test ANTHROPIC_BASE_URL is passed through as configured."""
        model, anthropic_cls, _ = self.build_model()

        anthropic_cls.assert_called_once_with(
            api_key='test-key', base_url='https://proxy.example/v1')
        self.assertEqual(model.base_url, 'https://proxy.example/v1')

    @patch.dict('os.environ',
                {
                    'ANTHROPIC_API_KEY': 'env-key',
                    'ANTHROPIC_BASE_URL': 'https://proxy.example',
                },
                clear=True)
    def test_env_key_is_resolved(self):
        """Test key='ENV' reads ANTHROPIC_API_KEY."""
        _, anthropic_cls, _ = self.build_model(key='ENV')

        anthropic_cls.assert_called_once_with(
            api_key='env-key', base_url='https://proxy.example')

    @patch.dict('os.environ', {}, clear=True)
    def test_system_prompt_is_top_level_parameter(self):
        """Test SYSTEM PromptList entries are not sent as message roles."""
        model, _, client = self.build_model(base_url='https://proxy')
        prompt = PromptList([
            {
                'role': 'SYSTEM',
                'prompt': 'system one'
            },
            {
                'role': 'HUMAN',
                'prompt': 'hello'
            },
            {
                'role': 'BOT',
                'prompt': 'hi'
            },
            {
                'role': 'SYSTEM',
                'prompt': 'system two'
            },
        ])

        self.assertEqual(model._generate(prompt, max_out_len=64), 'OK')
        api_params = client.messages.create.call_args.kwargs

        self.assertEqual(api_params['system'], 'system one\nsystem two')
        self.assertEqual(api_params['messages'], [
            {
                'role': 'user',
                'content': 'hello'
            },
            {
                'role': 'assistant',
                'content': 'hi'
            },
        ])

    @patch.dict('os.environ', {}, clear=True)
    def test_raw_prompt_roles_are_converted(self):
        """Test raw prompt system/user/assistant roles are supported."""
        model, _, client = self.build_model()
        prompt = PromptList([
            {
                'role': 'system',
                'content': 'raw system'
            },
            {
                'role': 'user',
                'content': 'raw user'
            },
            {
                'role': 'assistant',
                'content': 'raw assistant'
            },
        ])

        self.assertEqual(model._generate(prompt, max_out_len=64), 'OK')
        api_params = client.messages.create.call_args.kwargs

        self.assertEqual(api_params['system'], 'raw system')
        self.assertEqual(api_params['messages'], [
            {
                'role': 'user',
                'content': 'raw user'
            },
            {
                'role': 'assistant',
                'content': 'raw assistant'
            },
        ])

    @patch.dict('os.environ', {}, clear=True)
    def test_raw_prompt_plain_list_is_supported(self):
        """Test RawPromptTemplate list output can be sent directly."""
        model, _, client = self.build_model()
        prompt = [
            {
                'role': 'system',
                'content': 'raw system'
            },
            {
                'role': 'user',
                'content': 'raw user'
            },
        ]

        self.assertEqual(model._generate(prompt, max_out_len=64), 'OK')
        api_params = client.messages.create.call_args.kwargs

        self.assertEqual(api_params['system'], 'raw system')
        self.assertEqual(api_params['messages'], [{
            'role': 'user',
            'content': 'raw user'
        }])

    @patch.dict('os.environ', {}, clear=True)
    def test_thinking_does_not_force_streaming(self):
        """Test thinking requests remain non-streaming by default."""
        thinking = {'type': 'enabled', 'budget_tokens': 1024}
        model, _, client = self.build_model(thinking=thinking)

        self.assertEqual(model._generate('hello', max_out_len=64), 'OK')
        api_params = client.messages.create.call_args.kwargs

        self.assertEqual(api_params['thinking'], thinking)
        self.assertNotIn('stream', api_params)

    @patch.dict('os.environ', {}, clear=True)
    def test_generate_with_thinking_content(self):
        """Test non-streaming thinking content is preserved."""
        thinking = {'type': 'enabled', 'budget_tokens': 1024}
        model, _, client = self.build_model(thinking=thinking)
        client.messages.create.return_value = types.SimpleNamespace(content=[
            types.SimpleNamespace(type='thinking', thinking='Thinking process'),
            types.SimpleNamespace(type='text', text='Final answer'),
        ])

        self.assertEqual(model._generate('hello', max_out_len=64),
                         'Thinking process</think>Final answer')

    @patch.dict('os.environ', {}, clear=True)
    def test_generate_with_dict_thinking_content(self):
        """Test dict response blocks with thinking content are supported."""
        model, _, client = self.build_model()
        client.messages.create.return_value = types.SimpleNamespace(content=[
            {
                'type': 'thinking',
                'thinking': 'Thinking process'
            },
            {
                'type': 'text',
                'text': 'Final answer'
            },
        ])

        self.assertEqual(model._generate('hello', max_out_len=64),
                         'Thinking process</think>Final answer')

    @patch.dict('os.environ', {}, clear=True)
    def test_claude_extra_kwargs_are_merged_into_request(self):
        """Test claude_extra_kwargs are passed to Anthropic SDK requests."""
        extra_kwargs = {
            'metadata': {
                'benchmark': 'opencompass'
            },
            'extra_headers': {
                'anthropic-beta': 'test-beta'
            },
            'temperature': 0.7,
        }
        model, _, client = self.build_model(
            temperature=0.0, claude_extra_kwargs=extra_kwargs)

        self.assertEqual(model._generate('hello', max_out_len=64), 'OK')
        api_params = client.messages.create.call_args.kwargs

        self.assertEqual(api_params['metadata'], {'benchmark': 'opencompass'})
        self.assertEqual(api_params['extra_headers'],
                         {'anthropic-beta': 'test-beta'})
        self.assertEqual(api_params['temperature'], 0.7)

    @patch.dict('os.environ', {}, clear=True)
    def test_stream_true_uses_streaming_response(self):
        """Test stream=True accumulates thinking and text deltas."""
        model, _, client = self.build_model(stream=True)
        client.messages.create.return_value = iter([
            types.SimpleNamespace(type='message_start'),
            types.SimpleNamespace(
                type='content_block_delta',
                delta=types.SimpleNamespace(type='text_delta', text='HE'),
            ),
            types.SimpleNamespace(
                type='content_block_delta',
                delta=types.SimpleNamespace(type='text_delta', text='LLO'),
            ),
            types.SimpleNamespace(
                type='content_block_delta',
                delta=types.SimpleNamespace(type='thinking_delta',
                                            thinking='Thinking process'),
            ),
            types.SimpleNamespace(type='message_stop'),
        ])

        self.assertEqual(model._generate('hello', max_out_len=64),
                         'Thinking process</think>HELLO')
        api_params = client.messages.create.call_args.kwargs

        self.assertTrue(api_params['stream'])

    @patch.dict('os.environ', {}, clear=True)
    def test_generate_multiple_inputs_uses_progress_bar(self):
        """Test batched generation displays the same progress bar as OpenAI."""
        model, _, _ = self.build_model()

        with patch('opencompass.models.claude_sdk_api.tqdm') as mock_tqdm:
            mock_tqdm.side_effect = lambda iterable, **kwargs: iterable

            self.assertEqual(model.generate(['hello', 'world'], 64),
                             ['OK', 'OK'])

        tqdm_kwargs = mock_tqdm.call_args.kwargs
        self.assertEqual(tqdm_kwargs['total'], 2)
        self.assertEqual(tqdm_kwargs['desc'], 'Inferencing')

    @patch.dict('os.environ', {}, clear=True)
    def test_generate_single_input_skips_progress_bar(self):
        """Test single-input generation follows OpenAI's direct path."""
        model, _, _ = self.build_model()

        with patch('opencompass.models.claude_sdk_api.tqdm') as mock_tqdm:
            self.assertEqual(model.generate(['hello'], 64), ['OK'])

        mock_tqdm.assert_not_called()


if __name__ == '__main__':
    unittest.main()
