import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from opencompass.models.qwen_api import Qwen


def make_qwen_model(call_mock, retry=1):
    fake_dashscope = SimpleNamespace(
        Generation=SimpleNamespace(call=call_mock))
    with patch.dict(sys.modules, {'dashscope': fake_dashscope}):
        return Qwen(
            path='qwen3-max',
            key='test-key',
            query_per_second=1000,
            retry=retry,
        )


class TestQwenAPI(unittest.TestCase):

    def test_generate_extracts_output_text(self):
        response = SimpleNamespace(
            status_code=200,
            output=SimpleNamespace(text='Generated response'),
        )
        call_mock = MagicMock(return_value=response)
        model = make_qwen_model(call_mock)

        result = model._generate('Hello', max_out_len=16)

        self.assertEqual(result, 'Generated response')

    def test_generate_extracts_choice_message_content(self):
        response = SimpleNamespace(
            status_code=200,
            output=SimpleNamespace(
                text=None,
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='Final answer'))
                ],
            ),
        )
        call_mock = MagicMock(return_value=response)
        model = make_qwen_model(call_mock)

        result = model._generate('Hello', max_out_len=16)

        self.assertEqual(result, 'Final answer')

    @patch('opencompass.models.qwen_api.time.sleep', return_value=None)
    def test_generate_retries_when_success_response_has_no_text(self, _):
        empty_response = SimpleNamespace(
            status_code=200,
            output=SimpleNamespace(text=None, choices=[]),
        )
        success_response = SimpleNamespace(
            status_code=200,
            output=SimpleNamespace(text='After retry'),
        )
        call_mock = MagicMock(side_effect=[empty_response, success_response])
        model = make_qwen_model(call_mock, retry=2)

        result = model._generate('Hello', max_out_len=16)

        self.assertEqual(result, 'After retry')
        self.assertEqual(call_mock.call_count, 2)


if __name__ == '__main__':
    unittest.main()
