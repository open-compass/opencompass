import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from opencompass.models.turbomind_api import TurboMindAPIModel


class FakeAPIClient:

    def __init__(self):
        self.headers = {}
        self.calls = []

    def completions_v1(self, **kwargs):
        self.calls.append(kwargs)
        yield {'choices': [{'text': 'OK'}]}


class RetryAPIClient(FakeAPIClient):

    def __init__(self, failures):
        super().__init__()
        self.failures = failures

    def completions_v1(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.failures:
            yield {'choices': [{'text': 'partial'}]}
            raise ConnectionError('temporary connection error')
        yield {'choices': [{'text': 'OK'}]}


class FakeResponse:

    def __init__(self, ok=True, status_code=200, payload=None, text=''):
        self.ok = ok
        self.status_code = status_code
        self.text = text if text else str(payload)
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


class FlakyPost:
    """Fake requests.post failing the first ``failures`` calls."""

    def __init__(self, failures, payload):
        self.failures = failures
        self.payload = payload
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if len(self.calls) <= self.failures:
            return FakeResponse(ok=False,
                                status_code=502,
                                text='temporary bad gateway')
        return FakeResponse(payload=self.payload)


def make_turbomind_api_model(client, **kwargs):
    api_client_cls = MagicMock(return_value=client)
    fake_lmdeploy = SimpleNamespace()
    fake_serve = SimpleNamespace()
    fake_openai = SimpleNamespace()
    fake_api_client = SimpleNamespace(APIClient=api_client_cls)
    with patch.dict(
            sys.modules, {
                'lmdeploy': fake_lmdeploy,
                'lmdeploy.serve': fake_serve,
                'lmdeploy.serve.openai': fake_openai,
                'lmdeploy.serve.openai.api_client': fake_api_client,
            }):
        model = TurboMindAPIModel(
            model_name='test-model',
            api_addr='http://127.0.0.1:23333',
            **kwargs,
        )
    return model


class TestTurboMindAPIModel(unittest.TestCase):

    def test_default_generation_kwargs_keep_legacy_sampling(self):
        client = FakeAPIClient()
        model = make_turbomind_api_model(client)

        result = model._generate('Hello', max_out_len=16, temperature=0.7,
                                 end_str=None)

        self.assertEqual(result, 'OK')
        self.assertEqual(client.calls[0]['model'], 'test-model')
        self.assertEqual(client.calls[0]['max_tokens'], 16)
        self.assertEqual(client.calls[0]['temperature'], 0.7)
        self.assertEqual(client.calls[0]['top_p'], 0.8)
        self.assertEqual(client.calls[0]['top_k'], 50)
        self.assertIn('session_id', client.calls[0])

    def test_gen_config_is_forwarded_to_lmdeploy_client(self):
        client = FakeAPIClient()
        model = make_turbomind_api_model(
            client,
            gen_config=dict(
                max_new_tokens=23,
                random_seed=42,
                temperature=0.6,
                top_p=0.95,
                top_k=50,
            ),
        )

        result = model._generate('Hello', max_out_len=16, temperature=0.7,
                                 end_str=None)

        self.assertEqual(result, 'OK')
        self.assertEqual(client.calls[0]['max_tokens'], 23)
        self.assertEqual(client.calls[0]['temperature'], 0.6)
        self.assertEqual(client.calls[0]['top_p'], 0.95)
        self.assertEqual(client.calls[0]['top_k'], 50)
        self.assertEqual(client.calls[0]['random_seed'], 42)

    def test_constructor_top_k_override_is_forwarded(self):
        client = FakeAPIClient()
        model = make_turbomind_api_model(client, top_k=1, top_p=1.0)

        result = model._generate('Hello', max_out_len=16, temperature=0.7,
                                 end_str=None)

        self.assertEqual(result, 'OK')
        self.assertEqual(client.calls[0]['top_p'], 1.0)
        self.assertEqual(client.calls[0]['top_k'], 1)

    def test_generate_retries_and_discards_partial_response(self):
        client = RetryAPIClient(failures=1)
        model = make_turbomind_api_model(client, retry=2)

        result = model._generate('Hello', max_out_len=16, temperature=0.7,
                                 end_str=None)

        self.assertEqual(result, 'OK')
        self.assertEqual(len(client.calls), 2)

    def test_generate_raises_after_retry_exhausted(self):
        client = RetryAPIClient(failures=2)
        model = make_turbomind_api_model(client, retry=2)

        with self.assertRaisesRegex(RuntimeError,
                                    r'after retrying for 2 times'):
            model._generate('Hello', max_out_len=16, temperature=0.7,
                            end_str=None)

        self.assertEqual(len(client.calls), 2)

    def test_verbose_logs_each_api_output(self):
        client = FakeAPIClient()
        model = make_turbomind_api_model(client, verbose=True)
        model.logger = MagicMock()

        model._generate('Hello', max_out_len=16, temperature=0.7,
                        end_str=None)

        model.logger.info.assert_any_call('Start calling TurboMind API')
        model.logger.info.assert_any_call('TurboMind API output: %s',
                                          {'choices': [{'text': 'OK'}]})

    def test_verbose_disabled_does_not_log_api_output(self):
        client = FakeAPIClient()
        model = make_turbomind_api_model(client)
        model.logger = MagicMock()

        model._generate('Hello', max_out_len=16, temperature=0.7,
                        end_str=None)

        model.logger.info.assert_not_called()

    @patch('opencompass.models.turbomind_api.requests.post')
    def test_verbose_logs_encode_output(self, mock_post):
        mock_post.return_value = FakeResponse(
            payload={'input_ids': [1, 2, 3], 'length': 3})
        model = make_turbomind_api_model(FakeAPIClient(), verbose=True)
        model.logger = MagicMock()

        model.get_token_len('hello')

        model.logger.info.assert_any_call(
            'TurboMind API /v1/encode ok: input=%s, token_len=%d',
            "'hello'", 3)

    @patch('opencompass.models.turbomind_api.requests.post')
    def test_verbose_logs_encode_full_input(self, mock_post):
        mock_post.return_value = FakeResponse(
            payload={'input_ids': [1], 'length': 1})
        model = make_turbomind_api_model(FakeAPIClient(), verbose=True)
        model.logger = MagicMock()

        long_input = 'Passage:\nline1\nline2\n' + 'x' * 200
        model.get_token_len(long_input)

        model.logger.info.assert_any_call(
            'TurboMind API /v1/encode ok: input=%s, token_len=%d',
            repr(long_input), 1)

    @patch('opencompass.models.turbomind_api.requests.post')
    def test_verbose_logs_get_ppl_output(self, mock_post):
        mock_post.return_value = FakeResponse(payload={'ppl': 2.5})
        model = make_turbomind_api_model(FakeAPIClient(), verbose=True)
        model.logger = MagicMock()

        model._get_ppl('hello')

        model.logger.info.assert_any_call(
            'TurboMind API /get_ppl ok: input=%s, ppl=%s', "'hello'", 2.5)

    @patch('opencompass.models.turbomind_api.requests.post')
    def test_verbose_logs_get_ppl_token_ids_input(self, mock_post):
        mock_post.return_value = FakeResponse(payload={'ppl': 1.5})
        model = make_turbomind_api_model(FakeAPIClient(), verbose=True)
        model.logger = MagicMock()

        model._get_ppl([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        model.logger.info.assert_any_call(
            'TurboMind API /get_ppl ok: input=%s, ppl=%s',
            '10 token ids: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]', 1.5)

    @patch('opencompass.models.turbomind_api.requests.post')
    def test_verbose_disabled_does_not_log_encode_or_ppl(self, mock_post):
        mock_post.side_effect = [
            FakeResponse(payload={'input_ids': [1], 'length': 1}),
            FakeResponse(payload={'ppl': 2.0}),
        ]
        model = make_turbomind_api_model(FakeAPIClient())
        model.logger = MagicMock()

        model.get_token_len('hello')
        model._get_ppl('hello')

        model.logger.info.assert_not_called()

    @patch('opencompass.models.turbomind_api.requests.post')
    def test_encode_retries_and_succeeds(self, mock_post):
        flaky = FlakyPost(failures=1,
                          payload={'input_ids': [1, 2, 3], 'length': 3})
        mock_post.side_effect = flaky
        model = make_turbomind_api_model(FakeAPIClient(), retry=2)

        self.assertEqual(model.get_token_len('hello'), 3)
        self.assertEqual(len(flaky.calls), 2)

    @patch('opencompass.models.turbomind_api.requests.post')
    def test_encode_raises_after_retry_exhausted(self, mock_post):
        flaky = FlakyPost(failures=5,
                          payload={'input_ids': [1], 'length': 1})
        mock_post.side_effect = flaky
        model = make_turbomind_api_model(FakeAPIClient(), retry=2)

        with self.assertRaisesRegex(RuntimeError,
                                    r'/v1/encode failed after retrying '
                                    r'for 2 times'):
            model.get_token_len('hello')

        self.assertEqual(len(flaky.calls), 2)

    @patch('opencompass.models.turbomind_api.requests.post')
    def test_encode_unexpected_response_body_raises_with_detail(self,
                                                                mock_post):
        mock_post.return_value = FakeResponse(payload={'detail':
                                                       'Not Found'})
        model = make_turbomind_api_model(FakeAPIClient(), retry=1)

        with self.assertRaises(RuntimeError) as cm:
            model.get_token_len('hello')

        # The informative body lives in the chained cause error
        self.assertIn('unexpected response', str(cm.exception.__cause__))
        self.assertIn("{'detail': 'Not Found'}",
                      str(cm.exception.__cause__))

    @patch('opencompass.models.turbomind_api.requests.post')
    def test_get_ppl_retries_and_succeeds(self, mock_post):
        flaky = FlakyPost(failures=1, payload={'ppl': 2.5})
        mock_post.side_effect = flaky
        model = make_turbomind_api_model(FakeAPIClient(), retry=2)

        self.assertEqual(model._get_ppl('hello'), 2.5)
        self.assertEqual(len(flaky.calls), 2)

    @patch('opencompass.models.turbomind_api.requests.post')
    def test_get_ppl_raises_after_retry_exhausted(self, mock_post):
        flaky = FlakyPost(failures=5, payload={'ppl': 2.5})
        mock_post.side_effect = flaky
        model = make_turbomind_api_model(FakeAPIClient(), retry=2)

        with self.assertRaisesRegex(RuntimeError,
                                    r'/get_ppl failed after retrying '
                                    r'for 2 times'):
            model._get_ppl('hello')

        self.assertEqual(len(flaky.calls), 2)

    @patch('opencompass.models.turbomind_api.requests.post')
    def test_get_loglikelihood_retries_each_underlying_request(self,
                                                               mock_post):
        responses = [
            FakeResponse(payload={'input_ids': [1, 2], 'length': 2}),
            FakeResponse(payload={'input_ids': [1], 'length': 1}),
            FakeResponse(ok=False, status_code=502, text='boom'),
            FakeResponse(payload={'ppl': 4.0}),
            FakeResponse(payload={'ppl': 2.0}),
        ]
        mock_post.side_effect = responses
        model = make_turbomind_api_model(FakeAPIClient(), retry=2)

        # full ppl fails once then succeeds; context ppl succeeds directly
        result = model.get_loglikelihood(['context answer'], ['answer'])

        self.assertAlmostEqual(result[0], -(4.0 * 2 - 2.0 * 1))
        self.assertEqual(mock_post.call_count, 5)


if __name__ == '__main__':
    unittest.main()
