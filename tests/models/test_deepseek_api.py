import unittest
from unittest.mock import MagicMock, patch

import requests

from opencompass.models.deepseek_api import DeepseekAPI


class TestDeepseekAPI(unittest.TestCase):

    def _build_model(self, retry=2, timeout=12):
        model = DeepseekAPI(path='deepseek-reasoner',
                            key='test-key',
                            url='http://127.0.0.1:8000/v1/chat/completions',
                            retry=retry,
                            timeout=timeout)
        model.tokens = MagicMock()
        return model

    @patch('opencompass.models.deepseek_api.requests.request')
    def test_generate_passes_timeout_and_max_tokens(self, mock_request):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'answer'
                }
            }]
        }
        mock_request.return_value = mock_response

        model = self._build_model(timeout=12)
        result = model._generate('hello', max_out_len=321)

        self.assertEqual(result, 'answer')
        request_kwargs = mock_request.call_args.kwargs
        self.assertEqual(request_kwargs['timeout'], 12)
        self.assertEqual(request_kwargs['json']['max_tokens'], 321)
        model.tokens.acquire.assert_called_once()
        model.tokens.release.assert_called_once()

    @patch('opencompass.models.deepseek_api.time.sleep')
    @patch('opencompass.models.deepseek_api.requests.request')
    def test_request_exception_releases_token_and_stops_after_retry(
            self, mock_request, mock_sleep):
        mock_request.side_effect = requests.exceptions.Timeout('timeout')
        model = self._build_model(retry=2)

        with self.assertRaises(RuntimeError):
            model._generate('hello')

        self.assertEqual(mock_request.call_count, 2)
        self.assertEqual(model.tokens.acquire.call_count, 2)
        self.assertEqual(model.tokens.release.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch('opencompass.models.deepseek_api.time.sleep')
    @patch('opencompass.models.deepseek_api.requests.request')
    def test_429_response_stops_after_retry(self, mock_request, mock_sleep):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {'error': 'rate limited'}
        mock_request.return_value = mock_response
        model = self._build_model(retry=2)

        with self.assertRaises(RuntimeError):
            model._generate('hello')

        self.assertEqual(mock_request.call_count, 2)
        self.assertEqual(model.tokens.acquire.call_count, 2)
        self.assertEqual(model.tokens.release.call_count, 2)
        mock_sleep.assert_called_with(5)


if __name__ == '__main__':
    unittest.main()
