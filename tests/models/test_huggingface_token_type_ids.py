"""Tests for dropping token_type_ids before model forward in PPL paths."""

import unittest

import torch

from opencompass.models.huggingface_above_v4_33 import (
    HuggingFaceBaseModel, HuggingFacewithChatTemplate)


class _StopForward(Exception):
    pass


class _TokenTypeIdsTokenizer:
    """Mimics PreTrainedTokenizerFast, which also returns token_type_ids."""
    pad_token = '<pad>'
    pad_token_id = 0

    def batch_encode_plus(self, *args, **kwargs):
        return {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'token_type_ids': torch.zeros(1, 3, dtype=torch.long),
            'attention_mask': torch.ones(1, 3, dtype=torch.long),
        }


class _RecordingModel:
    device = 'cpu'

    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        raise _StopForward


class TestPPLTokenTypeIds(unittest.TestCase):

    def _assert_forward_skips_token_type_ids(self, model):
        self.assertNotIn('token_type_ids', model.model.kwargs)
        self.assertIn('input_ids', model.model.kwargs)
        self.assertIn('attention_mask', model.model.kwargs)

    def test_base_model_get_ppl_skips_token_type_ids(self):
        model = HuggingFaceBaseModel.__new__(HuggingFaceBaseModel)
        model.tokenizer = _TokenTypeIdsTokenizer()
        model.model = _RecordingModel()
        model.max_seq_len = 32
        model.drop_middle = False

        with self.assertRaises(_StopForward):
            model.get_ppl(['hello'])

        self._assert_forward_skips_token_type_ids(model)

    def test_chat_template_get_ppl_tokenwise_skips_token_type_ids(self):
        model = HuggingFacewithChatTemplate.__new__(
            HuggingFacewithChatTemplate)
        model.tokenizer = _TokenTypeIdsTokenizer()
        model.model = _RecordingModel()
        model.max_seq_len = 32

        with self.assertRaises(_StopForward):
            model.get_ppl_tokenwise(['hello'], [[(0, 1, 1)]])

        self._assert_forward_skips_token_type_ids(model)


if __name__ == '__main__':
    unittest.main()
