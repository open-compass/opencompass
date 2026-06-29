"""Unit tests for LiteLLMAPI."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from opencompass.models.litellm_api import LiteLLMAPI
from opencompass.utils.prompt import PromptList


def _install_litellm_stub():
    """Register a fake ``litellm`` module so ``import litellm`` resolves.

    Returns the stubbed ``completion`` MagicMock so tests can assert on calls
    and configure return values / side_effects.
    """
    fake = types.ModuleType('litellm')
    fake.completion = MagicMock(name='litellm.completion')
    sys.modules['litellm'] = fake
    return fake.completion


def _fake_response(content='hi'):
    """Build a minimal OpenAI-shaped ``ModelResponse`` stand-in."""
    message = SimpleNamespace(content=content, role='assistant')
    choice = SimpleNamespace(message=message, finish_reason='stop', index=0)
    return SimpleNamespace(choices=[choice], id='cmpl-test', model='test')


class TestLiteLLMAPIInit(unittest.TestCase):
    """Initialization, registration, and kwarg forwarding."""

    def test_registers_in_models_registry(self):
        from opencompass.registry import MODELS

        cls = MODELS.get('LiteLLMAPI')
        self.assertIs(cls, LiteLLMAPI)

    def test_default_init_stores_path(self):
        model = LiteLLMAPI(path='openai/gpt-4o-mini')
        self.assertEqual(model.path, 'openai/gpt-4o-mini')
        self.assertEqual(model.retry, 2)
        self.assertIsNone(model.key)
        self.assertIsNone(model.api_base)
        self.assertIsNone(model.api_version)

    def test_call_kwargs_forwards_provider_credentials(self):
        model = LiteLLMAPI(
            path='azure/gpt-4o',
            key='sk-test',
            api_base='https://example.azure.com',
            api_version='2025-01-01-preview',
            temperature=0.1,
            extra_body={'reasoning_effort': 'high'},
        )
        kwargs = model._build_call_kwargs(
            messages=[{'role': 'user', 'content': 'hi'}],
            max_out_len=128,
        )
        self.assertEqual(kwargs['model'], 'azure/gpt-4o')
        self.assertEqual(kwargs['api_key'], 'sk-test')
        self.assertEqual(kwargs['api_base'], 'https://example.azure.com')
        self.assertEqual(kwargs['api_version'], '2025-01-01-preview')
        self.assertEqual(kwargs['max_tokens'], 128)
        self.assertEqual(kwargs['temperature'], 0.1)
        self.assertEqual(kwargs['reasoning_effort'], 'high')

    def test_call_kwargs_omits_optional_when_blank(self):
        """When no creds are configured, LiteLLM must be allowed to resolve
        provider-specific env vars itself — we do not inject ``api_key=None``
        etc. which would cause 401s."""
        model = LiteLLMAPI(path='openai/gpt-4o-mini')
        kwargs = model._build_call_kwargs(
            messages=[{'role': 'user', 'content': 'hi'}],
            max_out_len=128,
        )
        self.assertNotIn('api_key', kwargs)
        self.assertNotIn('api_base', kwargs)
        self.assertNotIn('api_version', kwargs)
        self.assertNotIn('temperature', kwargs)

    def test_drop_params_default_true(self):
        model = LiteLLMAPI(path='openai/gpt-4o-mini')
        kwargs = model._build_call_kwargs(
            messages=[{'role': 'user', 'content': 'hi'}],
            max_out_len=128,
        )
        self.assertTrue(kwargs['drop_params'])

    def test_extra_body_cannot_override_core_params(self):
        """extra_body should not override model, messages, or max_tokens."""
        model = LiteLLMAPI(
            path='openai/gpt-4o-mini',
            extra_body={'model': 'WRONG', 'max_tokens': 999},
        )
        kwargs = model._build_call_kwargs(
            messages=[{'role': 'user', 'content': 'hi'}],
            max_out_len=128,
        )
        self.assertEqual(kwargs['model'], 'openai/gpt-4o-mini')
        self.assertEqual(kwargs['max_tokens'], 128)


class TestLiteLLMAPIMessages(unittest.TestCase):
    """Input-to-OpenAI-messages translation."""

    def test_plain_string_input(self):
        model = LiteLLMAPI(path='openai/gpt-4o-mini')
        messages = model._build_messages('hello')
        self.assertEqual(messages, [{'role': 'user', 'content': 'hello'}])

    def test_opencompass_native_prompt_list(self):
        """HUMAN/BOT/SYSTEM with 'prompt' key -> user/assistant/system
        with 'content' key."""
        model = LiteLLMAPI(path='openai/gpt-4o-mini')
        prompt_list = PromptList([
            {
                'role': 'SYSTEM',
                'prompt': 'You are helpful.'
            },
            {
                'role': 'HUMAN',
                'prompt': 'Hi there'
            },
            {
                'role': 'BOT',
                'prompt': 'Hello!'
            },
            {
                'role': 'HUMAN',
                'prompt': 'What is 2+2?'
            },
        ])
        messages = model._build_messages(prompt_list)
        self.assertEqual(messages, [
            {
                'role': 'system',
                'content': 'You are helpful.'
            },
            {
                'role': 'user',
                'content': 'Hi there'
            },
            {
                'role': 'assistant',
                'content': 'Hello!'
            },
            {
                'role': 'user',
                'content': 'What is 2+2?'
            },
        ])

    def test_chatml_shaped_prompt_list_passes_through(self):
        """When upstream has already emitted OpenAI-style CHATML, the message
        list should be forwarded unchanged (modulo possible system_prompt
        prepending)."""
        model = LiteLLMAPI(path='openai/gpt-4o-mini')
        chatml = [
            {
                'role': 'system',
                'content': 'sys'
            },
            {
                'role': 'user',
                'content': 'u'
            },
        ]
        messages = model._build_messages(chatml)
        self.assertEqual(messages, chatml)

    def test_system_prompt_prepended_when_absent(self):
        model = LiteLLMAPI(path='openai/gpt-4o-mini',
                           system_prompt='Be concise.')
        messages = model._build_messages('hi')
        self.assertEqual(messages[0], {
            'role': 'system',
            'content': 'Be concise.'
        })
        self.assertEqual(messages[1], {'role': 'user', 'content': 'hi'})

    def test_system_prompt_not_duplicated_when_system_exists(self):
        """If the caller already supplied a system message, our
        ``system_prompt`` must not create a second one."""
        model = LiteLLMAPI(path='openai/gpt-4o-mini',
                           system_prompt='IGNORE ME')
        chatml = [
            {
                'role': 'system',
                'content': 'caller-supplied'
            },
            {
                'role': 'user',
                'content': 'hi'
            },
        ]
        messages = model._build_messages(chatml)
        system_messages = [m for m in messages if m['role'] == 'system']
        self.assertEqual(len(system_messages), 1)
        self.assertEqual(system_messages[0]['content'], 'caller-supplied')


class TestLiteLLMAPIGenerate(unittest.TestCase):
    """End-to-end ``generate`` tests with mocked LiteLLM."""

    def test_generate_single_string(self):
        completion = _install_litellm_stub()
        completion.return_value = _fake_response('pong')

        model = LiteLLMAPI(path='openai/gpt-4o-mini', key='sk-test')
        results = model.generate(['ping'], max_out_len=32)

        self.assertEqual(results, ['pong'])
        completion.assert_called_once()
        kwargs = completion.call_args.kwargs
        self.assertEqual(kwargs['model'], 'openai/gpt-4o-mini')
        self.assertEqual(kwargs['messages'],
                         [{
                             'role': 'user',
                             'content': 'ping'
                         }])
        self.assertEqual(kwargs['max_tokens'], 32)
        self.assertEqual(kwargs['api_key'], 'sk-test')

    def test_generate_preserves_batch_order(self):
        completion = _install_litellm_stub()
        completion.side_effect = [
            _fake_response('a'),
            _fake_response('b'),
            _fake_response('c'),
        ]

        # Force single-worker to keep deterministic order for the side_effect
        # sequence. (In production order is preserved by ``executor.map``
        # regardless of workers because we collect with ``list(...)``.)
        model = LiteLLMAPI(path='openai/gpt-4o-mini', max_workers=1)
        results = model.generate(['1', '2', '3'])

        self.assertEqual(results, ['a', 'b', 'c'])
        self.assertEqual(completion.call_count, 3)

    def test_generate_handles_none_content(self):
        """Some providers return ``message.content = None`` when output is
        filtered; we must return an empty string rather than raise."""
        completion = _install_litellm_stub()
        completion.return_value = _fake_response(content=None)

        model = LiteLLMAPI(path='openai/gpt-4o-mini')
        results = model.generate(['x'])
        self.assertEqual(results, [''])

    def test_generate_retries_then_succeeds(self):
        completion = _install_litellm_stub()
        completion.side_effect = [
            Exception('transient'),
            _fake_response('recovered'),
        ]

        model = LiteLLMAPI(path='openai/gpt-4o-mini', retry=3)
        with patch('time.sleep'):  # skip the backoff
            results = model.generate(['x'])
        self.assertEqual(results, ['recovered'])
        self.assertEqual(completion.call_count, 2)

    def test_generate_raises_after_exhausting_retries(self):
        completion = _install_litellm_stub()
        completion.side_effect = Exception('permanent')

        model = LiteLLMAPI(path='openai/gpt-4o-mini', retry=2)
        with patch('time.sleep'):
            with self.assertRaises(RuntimeError):
                model.generate(['x'])
        self.assertEqual(completion.call_count, 2)

    def test_generate_translates_opencompass_native_prompt_list(self):
        completion = _install_litellm_stub()
        completion.return_value = _fake_response('4')

        model = LiteLLMAPI(path='openai/gpt-4o-mini')
        prompt = PromptList([
            {
                'role': 'HUMAN',
                'prompt': 'What is 2+2?'
            },
        ])
        results = model.generate([prompt])

        self.assertEqual(results, ['4'])
        kwargs = completion.call_args.kwargs
        self.assertEqual(kwargs['messages'],
                         [{
                             'role': 'user',
                             'content': 'What is 2+2?'
                         }])

    def test_generate_raises_import_error_without_litellm(self):
        sys.modules.pop('litellm', None)
        with patch.dict(sys.modules, {'litellm': None}):
            model = LiteLLMAPI(path='openai/gpt-4o-mini', retry=1)
            with self.assertRaises(ImportError) as ctx:
                model.generate(['hi'])
            self.assertIn('pip install litellm', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
