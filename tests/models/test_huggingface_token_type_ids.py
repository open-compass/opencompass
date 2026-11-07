"""Tests for Hugging Face tokenizer outputs used by PPL paths."""

import unittest

from opencompass.models.huggingface_above_v4_33 import (
    HuggingFaceBaseModel, HuggingFacewithChatTemplate)


class _StopTokenization(Exception):
    pass


class _RecordingTokenizer:
    pad_token = '<pad>'
    pad_token_id = 0

    def __init__(self):
        self.kwargs = None

    def batch_encode_plus(self, *args, **kwargs):
        self.kwargs = kwargs
        raise _StopTokenization


class TestPPLTokenization(unittest.TestCase):

    def test_base_model_get_ppl_skips_token_type_ids(self):
        model = HuggingFaceBaseModel.__new__(HuggingFaceBaseModel)
        model.tokenizer = _RecordingTokenizer()
        model.max_seq_len = 32
        model.drop_middle = False

        with self.assertRaises(_StopTokenization):
            model.get_ppl(['hello'])

        self.assertFalse(model.tokenizer.kwargs['return_token_type_ids'])

    def test_chat_template_get_ppl_tokenwise_skips_token_type_ids(self):
        model = HuggingFacewithChatTemplate.__new__(
            HuggingFacewithChatTemplate)
        model.tokenizer = _RecordingTokenizer()
        model.max_seq_len = 32

        with self.assertRaises(_StopTokenization):
            model.get_ppl_tokenwise(['hello'], [[(0, 1, 1)]])

        self.assertFalse(model.tokenizer.kwargs['return_token_type_ids'])


if __name__ == '__main__':
    unittest.main()
