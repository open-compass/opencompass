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


if __name__ == '__main__':
    unittest.main()
