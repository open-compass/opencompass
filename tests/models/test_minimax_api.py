"""Unit tests for MiniMax API models."""

import unittest
from unittest.mock import MagicMock, patch

from opencompass.models.minimax_api import (
    MINIMAX_API_BASE,
    MiniMaxAPI,
    MiniMaxChatCompletionV2,
)


class TestMiniMaxAPI(unittest.TestCase):
    """Test cases for MiniMaxAPI."""

    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_initialization_basic(self):
        """Test basic initialization with ENV key."""
        model = MiniMaxAPI(
            path='MiniMax-M2.7',
            max_seq_len=204800,
            query_per_second=2,
        )

        self.assertEqual(model.path, 'MiniMax-M2.7')
        self.assertEqual(model.model, 'MiniMax-M2.7')
        self.assertEqual(model.max_seq_len, 204800)
        self.assertEqual(model.url, MINIMAX_API_BASE)
        self.assertEqual(model.keys, ['test-key'])
        self.assertIsNone(model.temperature)

    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'key1,key2,key3'})
    def test_initialization_multiple_keys(self):
        """Test initialization with multiple comma-separated keys."""
        model = MiniMaxAPI(path='MiniMax-M2.7')

        self.assertEqual(len(model.keys), 3)
        self.assertEqual(model.keys[0], 'key1')
        self.assertEqual(model.keys[1], 'key2')
        self.assertEqual(model.keys[2], 'key3')

    def test_initialization_with_direct_key(self):
        """Test initialization with a direct API key."""
        model = MiniMaxAPI(
            path='MiniMax-M2.5',
            key='my-direct-key',
        )

        self.assertEqual(model.keys, ['my-direct-key'])

    def test_initialization_with_key_list(self):
        """Test initialization with a list of keys."""
        model = MiniMaxAPI(
            path='MiniMax-M2.5',
            key=['key-a', 'key-b'],
        )

        self.assertEqual(len(model.keys), 2)
        self.assertEqual(model.keys[0], 'key-a')
        self.assertEqual(model.keys[1], 'key-b')

    @patch.dict('os.environ', {}, clear=True)
    def test_initialization_missing_env_key(self):
        """Test that missing MINIMAX_API_KEY raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            MiniMaxAPI(path='MiniMax-M2.7', key='ENV')
        self.assertIn('MINIMAX_API_KEY', str(ctx.exception))

    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_initialization_with_temperature(self):
        """Test initialization with temperature."""
        model = MiniMaxAPI(
            path='MiniMax-M2.7',
            temperature=0.7,
        )

        self.assertEqual(model.temperature, 0.7)

    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_initialization_with_custom_url(self):
        """Test initialization with custom URL."""
        custom_url = 'https://custom.minimax.io/v1/chat/completions'
        model = MiniMaxAPI(
            path='MiniMax-M2.7',
            url=custom_url,
        )

        self.assertEqual(model.url, custom_url)

    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_default_model(self):
        """Test that default model is MiniMax-M2.7."""
        model = MiniMaxAPI()

        self.assertEqual(model.model, 'MiniMax-M2.7')

    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_key_rotation(self):
        """Test round-robin key rotation."""
        model = MiniMaxAPI(
            path='MiniMax-M2.7',
            key=['key-1', 'key-2', 'key-3'],
        )

        headers1 = model._get_headers()
        headers2 = model._get_headers()
        headers3 = model._get_headers()
        headers4 = model._get_headers()

        self.assertEqual(headers1['Authorization'], 'Bearer key-1')
        self.assertEqual(headers2['Authorization'], 'Bearer key-2')
        self.assertEqual(headers3['Authorization'], 'Bearer key-3')
        # Wraps around
        self.assertEqual(headers4['Authorization'], 'Bearer key-1')

    @patch('opencompass.models.minimax_api.requests')
    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_generate_basic(self, mock_requests):
        """Test basic generate functionality."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Hello, world!'
                }
            }]
        }
        mock_requests.request.return_value = mock_response

        model = MiniMaxAPI(path='MiniMax-M2.7')
        results = model.generate(['Hi there'], max_out_len=100)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], 'Hello, world!')
        mock_requests.request.assert_called()

    @patch('opencompass.models.minimax_api.requests')
    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_generate_with_reasoning_content(self, mock_requests):
        """Test generate with reasoning_content in response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'The answer is 42.',
                    'reasoning_content': 'Let me think step by step...'
                }
            }]
        }
        mock_requests.request.return_value = mock_response

        model = MiniMaxAPI(
            path='MiniMax-M2.7',
            think_tag='</think>',
        )
        results = model.generate(['What is the meaning of life?'],
                                 max_out_len=200)

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0],
            'Let me think step by step...</think>The answer is 42.')

    @patch('opencompass.models.minimax_api.requests')
    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_generate_strips_inline_think_tags(self, mock_requests):
        """Test that inline <think>...</think> tags are stripped."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content':
                    '<think>internal reasoning here</think>Final answer'
                }
            }]
        }
        mock_requests.request.return_value = mock_response

        model = MiniMaxAPI(path='MiniMax-M2.7')
        results = model.generate(['Test'], max_out_len=100)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], 'Final answer')

    @patch('opencompass.models.minimax_api.requests')
    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_generate_with_temperature(self, mock_requests):
        """Test that temperature is passed in request data."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Response'
                }
            }]
        }
        mock_requests.request.return_value = mock_response

        model = MiniMaxAPI(
            path='MiniMax-M2.7',
            temperature=0.5,
        )
        model.generate(['Hello'], max_out_len=100)

        call_args = mock_requests.request.call_args
        sent_data = call_args[1]['json']
        self.assertEqual(sent_data['temperature'], 0.5)

    @patch('opencompass.models.minimax_api.requests')
    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_temperature_clamping_high(self, mock_requests):
        """Test that temperature > 1.0 is clamped to 1.0."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Response'
                }
            }]
        }
        mock_requests.request.return_value = mock_response

        model = MiniMaxAPI(
            path='MiniMax-M2.7',
            temperature=2.0,
        )
        model.generate(['Hello'], max_out_len=100)

        call_args = mock_requests.request.call_args
        sent_data = call_args[1]['json']
        self.assertEqual(sent_data['temperature'], 1.0)

    @patch('opencompass.models.minimax_api.requests')
    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_temperature_clamping_negative(self, mock_requests):
        """Test that negative temperature is clamped to 0."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Response'
                }
            }]
        }
        mock_requests.request.return_value = mock_response

        model = MiniMaxAPI(
            path='MiniMax-M2.7',
            temperature=-0.5,
        )
        model.generate(['Hello'], max_out_len=100)

        call_args = mock_requests.request.call_args
        sent_data = call_args[1]['json']
        self.assertEqual(sent_data['temperature'], 0)

    @patch('opencompass.models.minimax_api.requests')
    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_no_temperature_by_default(self, mock_requests):
        """Test that temperature is not included when not specified."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Response'
                }
            }]
        }
        mock_requests.request.return_value = mock_response

        model = MiniMaxAPI(path='MiniMax-M2.7')
        model.generate(['Hello'], max_out_len=100)

        call_args = mock_requests.request.call_args
        sent_data = call_args[1]['json']
        self.assertNotIn('temperature', sent_data)

    @patch('opencompass.models.minimax_api.requests')
    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_generate_with_system_prompt(self, mock_requests):
        """Test that system prompt is prepended."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Response'
                }
            }]
        }
        mock_requests.request.return_value = mock_response

        model = MiniMaxAPI(
            path='MiniMax-M2.7',
            system_prompt='You are a helpful assistant.',
        )
        model.generate(['Hello'], max_out_len=100)

        call_args = mock_requests.request.call_args
        sent_data = call_args[1]['json']
        self.assertEqual(sent_data['messages'][0]['role'], 'system')
        self.assertEqual(sent_data['messages'][0]['content'],
                         'You are a helpful assistant.')
        self.assertEqual(sent_data['messages'][1]['role'], 'user')

    @patch('opencompass.models.minimax_api.requests')
    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_generate_multiple_inputs(self, mock_requests):
        """Test generating with multiple inputs."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Generated'
                }
            }]
        }
        mock_requests.request.return_value = mock_response

        model = MiniMaxAPI(path='MiniMax-M2.7')
        results = model.generate(['Input 1', 'Input 2', 'Input 3'],
                                 max_out_len=100)

        self.assertEqual(len(results), 3)
        self.assertTrue(all(r == 'Generated' for r in results))

    @patch('opencompass.models.minimax_api.requests')
    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_generate_with_prompt_list(self, mock_requests):
        """Test generate with PromptList input."""
        from opencompass.utils.prompt import PromptList

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Response'
                }
            }]
        }
        mock_requests.request.return_value = mock_response

        model = MiniMaxAPI(path='MiniMax-M2.7')

        prompt_list = PromptList([
            {
                'role': 'HUMAN',
                'prompt': 'Hello'
            },
            {
                'role': 'BOT',
                'prompt': 'Hi there'
            },
            {
                'role': 'HUMAN',
                'prompt': 'How are you?'
            },
        ])

        results = model.generate([prompt_list], max_out_len=100)

        self.assertEqual(len(results), 1)
        call_args = mock_requests.request.call_args
        sent_data = call_args[1]['json']
        messages = sent_data['messages']
        self.assertEqual(messages[0]['role'], 'user')
        self.assertEqual(messages[1]['role'], 'assistant')
        self.assertEqual(messages[2]['role'], 'user')

    @patch('opencompass.models.minimax_api.requests')
    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_generate_rate_limit_retry(self, mock_requests):
        """Test retry on rate limit (base_resp status_code 1002)."""
        rate_limited = MagicMock()
        rate_limited.status_code = 200
        rate_limited.json.return_value = {
            'base_resp': {
                'status_code': 1002,
                'status_msg': 'rate limit'
            }
        }

        success = MagicMock()
        success.status_code = 200
        success.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Success after retry'
                }
            }]
        }

        mock_requests.request.side_effect = [rate_limited, success]

        model = MiniMaxAPI(path='MiniMax-M2.7', retry=3)
        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(results[0], 'Success after retry')

    @patch('opencompass.models.minimax_api.requests')
    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_generate_http_429_retry(self, mock_requests):
        """Test retry on HTTP 429 rate limit."""
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.json.return_value = {'error': 'rate limit'}

        success = MagicMock()
        success.status_code = 200
        success.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Success'
                }
            }]
        }

        mock_requests.request.side_effect = [rate_limited, success]

        model = MiniMaxAPI(path='MiniMax-M2.7', retry=3)
        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(results[0], 'Success')

    @patch('opencompass.models.minimax_api.requests')
    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_generate_all_retries_exhausted(self, mock_requests):
        """Test RuntimeError when all retries are exhausted."""
        fail = MagicMock()
        fail.status_code = 500
        fail.json.return_value = {'error': 'server error'}

        mock_requests.request.return_value = fail

        model = MiniMaxAPI(path='MiniMax-M2.7', retry=2)

        with self.assertRaises(RuntimeError):
            model.generate(['Hello'], max_out_len=100)

    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_api_base_url(self):
        """Test that the default API base URL is correct."""
        self.assertEqual(MINIMAX_API_BASE,
                         'https://api.minimax.io/v1/chat/completions')

    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'test-key'})
    def test_think_tag_default(self):
        """Test default think_tag is </think>."""
        model = MiniMaxAPI(path='MiniMax-M2.7')
        self.assertEqual(model.think_tag, '</think>')


class TestMiniMaxChatCompletionV2(unittest.TestCase):
    """Test cases for MiniMaxChatCompletionV2 (backward compatibility)."""

    def test_initialization(self):
        """Test basic initialization."""
        model = MiniMaxChatCompletionV2(
            path='MiniMax-M2.5',
            key='test-key',
        )

        self.assertEqual(model.model, 'MiniMax-M2.5')
        self.assertEqual(model.url, MINIMAX_API_BASE)

    def test_initialization_with_custom_url(self):
        """Test initialization with custom URL."""
        model = MiniMaxChatCompletionV2(
            path='abab5.5-chat',
            key='test-key',
            url='https://api.minimax.chat/v1/text/chatcompletion_v2',
        )

        self.assertEqual(
            model.url,
            'https://api.minimax.chat/v1/text/chatcompletion_v2')

    @patch('opencompass.models.minimax_api.requests')
    def test_generate_basic(self, mock_requests):
        """Test basic generate with MiniMaxChatCompletionV2."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Hello from MiniMax'
                }
            }]
        }
        mock_requests.request.return_value = mock_response

        model = MiniMaxChatCompletionV2(
            path='MiniMax-M2.5',
            key='test-key',
        )
        results = model.generate(['Hi'], max_out_len=100)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], 'Hello from MiniMax')


class TestMiniMaxAPIIntegration(unittest.TestCase):
    """Integration tests for MiniMaxAPI (require MINIMAX_API_KEY)."""

    @unittest.skipUnless(
        __import__('os').environ.get('MINIMAX_API_KEY'),
        'MINIMAX_API_KEY not set')
    def test_real_api_call(self):
        """Test a real API call to MiniMax."""
        model = MiniMaxAPI(
            path='MiniMax-M2.7',
            key='ENV',
            retry=2,
        )

        results = model.generate(['Say hello'],
                                 max_out_len=200)
        self.assertEqual(len(results), 1)
        self.assertTrue(
            'hello' in results[0].lower() or 'hi' in results[0].lower())

    @unittest.skipUnless(
        __import__('os').environ.get('MINIMAX_API_KEY'),
        'MINIMAX_API_KEY not set')
    def test_real_api_m2_5_highspeed(self):
        """Test a real API call with MiniMax-M2.5-highspeed."""
        model = MiniMaxAPI(
            path='MiniMax-M2.5-highspeed',
            key='ENV',
            retry=2,
        )

        # Use higher max_out_len since M2.5-highspeed includes
        # inline <think>...</think> reasoning tokens
        results = model.generate(['What is 2+2? Answer with just the number.'],
                                 max_out_len=500)
        self.assertEqual(len(results), 1)
        self.assertIn('4', results[0])

    @unittest.skipUnless(
        __import__('os').environ.get('MINIMAX_API_KEY'),
        'MINIMAX_API_KEY not set')
    def test_real_api_with_temperature(self):
        """Test a real API call with temperature setting."""
        model = MiniMaxAPI(
            path='MiniMax-M2.7',
            key='ENV',
            temperature=0.1,
            retry=2,
        )

        results = model.generate(
            ['Capital of France? One word.'],
            max_out_len=200)
        self.assertEqual(len(results), 1)


if __name__ == '__main__':
    unittest.main()
