"""Unit tests for ClaudeSDK."""

import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from opencompass.models.claude_sdk_api import ClaudeSDK
from opencompass.utils import get_logger
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

    @patch.dict('os.environ', {}, clear=True)
    def test_extra_headers_are_passed_to_client(self):
        """Test extra_headers are passed to the Anthropic SDK client."""
        headers = {
            'anthropic-beta': 'prompt-caching-2024-07-31',
            'x-provider-region': 'test-region',
        }
        _, anthropic_cls, _ = self.build_model(
            base_url='https://proxy.example/v1', extra_headers=headers)

        anthropic_cls.assert_called_once_with(
            api_key='test-key',
            base_url='https://proxy.example/v1',
            default_headers=headers)

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
    def test_temperature_is_never_sent(self):
        """Test temperature is left out: the SDK and API reject it."""
        for kwargs in ({}, {
                'temperature': 0.3
        }, {
                'temperature': None
        }, {
                'thinking': {
                    'type': 'enabled',
                    'budget_tokens': 1024
                }
        }):
            model, _, client = self.build_model(**kwargs)

            model._generate('hello', max_out_len=64)

            api_params = client.messages.create.call_args.kwargs
            self.assertNotIn('temperature', api_params)

    @patch.dict('os.environ', {}, clear=True)
    def test_configured_temperature_logs_a_warning(self):
        """Test a configured temperature is reported as ignored."""
        with self.assertLogs(get_logger(), level='WARNING') as logs:
            self.build_model(temperature=0.3)
        self.assertTrue(
            any('ignores temperature' in line for line in logs.output))

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
            types.SimpleNamespace(
                type='message_delta',
                delta=types.SimpleNamespace(stop_reason='end_turn'),
            ),
            types.SimpleNamespace(type='message_stop'),
        ])

        self.assertEqual(model._generate('hello', max_out_len=64),
                         'Thinking process</think>HELLO')
        api_params = client.messages.create.call_args.kwargs

        self.assertTrue(api_params['stream'])

    @staticmethod
    def stream_events(text, stop_reason=None, as_dict=False):
        """Build stream events, truncated when stop_reason is None."""
        deltas = [('content_block_delta', dict(thinking='Thinking')),
                  ('content_block_delta', dict(text=text))]
        if stop_reason is not None:
            deltas.append(('message_delta', dict(stop_reason=stop_reason)))
        if as_dict:
            return iter([dict(type=t, delta=d) for t, d in deltas])
        return iter([
            types.SimpleNamespace(type=t, delta=types.SimpleNamespace(**d))
            for t, d in deltas
        ])

    @patch.dict('os.environ', {}, clear=True)
    def test_stream_without_stop_reason_is_retried(self):
        """Test a truncated stream is dropped and requested again."""
        for as_dict in (False, True):
            with self.subTest(as_dict=as_dict):
                model, _, client = self.build_model(stream=True)
                model.retry = 2
                client.messages.create.side_effect = [
                    self.stream_events('par', as_dict=as_dict),
                    self.stream_events('full', 'end_turn', as_dict=as_dict),
                ]

                self.assertEqual(model._generate('hello', max_out_len=64),
                                 'Thinking</think>full')
                self.assertEqual(client.messages.create.call_count, 2)

    @patch.dict('os.environ', {}, clear=True)
    def test_stream_without_stop_reason_fails_after_retries(self):
        """Test a truncated reply is never returned as the prediction."""
        model, _, client = self.build_model(stream=True)
        model.retry = 2
        client.messages.create.side_effect = [
            self.stream_events('par'),
            self.stream_events('par'),
        ]

        with self.assertRaises(RuntimeError):
            model._generate('hello', max_out_len=64)
        self.assertEqual(client.messages.create.call_count, 2)

    @patch.dict('os.environ', {}, clear=True)
    def test_stream_max_tokens_is_not_retried(self):
        """Test a reply that used up max_tokens counts as finished."""
        model, _, client = self.build_model(stream=True)
        model.retry = 2
        client.messages.create.side_effect = [
            self.stream_events('par', 'max_tokens')
        ]

        self.assertEqual(model._generate('hello', max_out_len=64),
                         'Thinking</think>par')
        self.assertEqual(client.messages.create.call_count, 1)

    @patch.dict('os.environ', {}, clear=True)
    def test_stop_reason_confirm_can_be_disabled(self):
        """Test stop_reason_confirm=False keeps the truncated reply."""
        model, _, client = self.build_model(stream=True,
                                            stop_reason_confirm=False)
        client.messages.create.side_effect = [self.stream_events('par')]

        self.assertEqual(model._generate('hello', max_out_len=64),
                         'Thinking</think>par')

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

    @patch.dict('os.environ', {}, clear=True)
    def test_max_workers_sizes_the_thread_pool(self):
        """Test max_workers is used for batched generation threads."""
        model, _, _ = self.build_model(max_workers=3)

        with patch('opencompass.models.claude_sdk_api.ThreadPoolExecutor',
                   wraps=ThreadPoolExecutor) as mock_pool:
            self.assertEqual(model.generate(['hello', 'world'], 64),
                             ['OK', 'OK'])

        self.assertEqual(model.max_workers, 3)
        self.assertEqual(mock_pool.call_args.kwargs['max_workers'], 3)
        self.assertEqual(self.build_model()[0].max_workers, 32)

    @patch.dict('os.environ', {}, clear=True)
    def test_concurrent_tokens_cap_in_flight_requests(self):
        """Test the semaphore injected by concurrent infer caps requests."""
        model, _, client = self.build_model()
        model.tokens = threading.Semaphore(2)
        lock = threading.Lock()
        state = {'current': 0, 'peak': 0}

        def create(**kwargs):
            with lock:
                state['current'] += 1
                state['peak'] = max(state['peak'], state['current'])
            time.sleep(0.05)
            with lock:
                state['current'] -= 1
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(type='text', text='OK')])

        client.messages.create.side_effect = create
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(
                executor.map(lambda _: model._generate('hi', 16), range(8)))

        self.assertEqual(results, ['OK'] * 8)
        self.assertEqual(state['peak'], 2)

    @patch.dict('os.environ', {}, clear=True)
    def test_failed_request_releases_concurrent_token(self):
        """Test a failed request gives its semaphore slot back."""
        model, _, client = self.build_model()
        model.tokens = threading.Semaphore(1)
        client.messages.create.side_effect = RuntimeError('boom')

        with self.assertRaises(RuntimeError):
            model._generate('hi', 16)
        self.assertTrue(model.tokens.acquire(blocking=False))


if __name__ == '__main__':
    unittest.main()
