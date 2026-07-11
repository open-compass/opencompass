# docformatter: noqa
"""Unit tests for HuggingFace."""

import unittest
from unittest.mock import MagicMock, patch

from opencompass.models.huggingface import HuggingFace


class _ReadOnlyPadTokenizer:
    """Mimic ChatGLM3's ChatGLMTokenizer whose ``pad_token_id`` is a
    read-only ``@property`` without a setter. See issue #725.
    """

    vocab_size = 65024

    @property
    def pad_token_id(self):
        # Returns a value different from what the user requested, so the
        # "not consistent" warning branch in _load_tokenizer is exercised.
        return 0


class TestHuggingFace(unittest.TestCase):
    """Test cases for HuggingFace."""

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    def test_initialization_basic(self, mock_model_class,
                                  mock_tokenizer_class):
        """Test basic initialization."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model = MagicMock()
        mock_model.device = 'cpu'
        mock_model_class.from_pretrained.return_value = mock_model

        model = HuggingFace(
            path='test/model/path',
            max_seq_len=2048,
            model_kwargs=dict(device_map='cpu'),
            generation_kwargs=dict(temperature=0.7),
        )

        self.assertEqual(model.path, 'test/model/path')
        self.assertEqual(model.max_seq_len, 2048)
        self.assertEqual(model.generation_kwargs, dict(temperature=0.7))
        self.assertFalse(model.tokenizer_only)
        self.assertEqual(model.mode, 'none')

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    @patch('opencompass.models.huggingface.MultiTokenEOSCriteria')
    @patch('opencompass.models.huggingface.transformers')
    @patch('opencompass.models.huggingface.torch')
    def test_generate_basic(self, mock_torch, mock_transformers,
                            mock_stopping_criteria_class, mock_model_class,
                            mock_tokenizer_class):
        """Test basic generate functionality."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token = None
        mock_tokenizer.encode.return_value = [1, 2, 3]
        # Mock tokenizer call to return input_ids
        mock_tokenizer.return_value = {'input_ids': [[1, 2, 3]]}
        mock_tokenizer.batch_decode.return_value = ['Generated response']
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        # Mock torch.tensor to return a mock tensor
        mock_input_tensor = MagicMock()
        mock_input_tensor.shape = [1, 3]  # batch_size=1, seq_len=3
        mock_torch.tensor.return_value = mock_input_tensor
        mock_model = MagicMock()
        mock_model.device = 'cpu'
        # Mock output tensor that supports slicing
        mock_output = MagicMock()
        mock_output.shape = (1, 5)
        # Support slicing: outputs[:, input_ids.shape[1]:]
        mock_output.__getitem__.return_value = mock_output
        mock_model.generate.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model

        mock_stopping_criteria = MagicMock()
        mock_stopping_criteria_class.return_value = mock_stopping_criteria
        mock_stopping_criteria_list = MagicMock()
        mock_transformers.StoppingCriteriaList.return_value = mock_stopping_criteria_list  # noqa

        model = HuggingFace(
            path='test/model/path',
            max_seq_len=2048,
            model_kwargs=dict(device_map='cpu'),
            generation_kwargs=dict(temperature=0.7),
        )

        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], 'Generated response')
        mock_model.generate.assert_called()

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    @patch('opencompass.models.huggingface.torch')
    def test_generate_uses_max_new_tokens_when_max_out_len_is_none(
            self, mock_torch, mock_model_class, mock_tokenizer_class):
        """Test max_new_tokens is used when max_out_len is None."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token = None
        mock_tokenizer.return_value = {'input_ids': [[1, 2, 3]]}
        mock_tokenizer.batch_decode.return_value = ['Generated response']
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_input_tensor = MagicMock()
        mock_input_tensor.shape = [1, 3]
        mock_torch.tensor.return_value = mock_input_tensor

        mock_model = MagicMock()
        mock_model.device = 'cpu'
        mock_output = MagicMock()
        mock_output.__getitem__.return_value = mock_output
        mock_model.generate.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model

        model = HuggingFace(
            path='test/model/path',
            max_seq_len=2048,
            model_kwargs=dict(device_map='cpu'),
            generation_kwargs=dict(max_new_tokens=32, temperature=0.0),
        )

        results = model.generate(['Hello'], max_out_len=None)

        self.assertEqual(results, ['Generated response'])
        mock_tokenizer.assert_called_with(['Hello'],
                                          truncation=True,
                                          max_length=2016)

        generate_kwargs = mock_model.generate.call_args.kwargs
        self.assertEqual(generate_kwargs['max_new_tokens'], 32)
        self.assertEqual(generate_kwargs['temperature'], 0.0)

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    def test_generate_requires_length_when_max_out_len_is_none(
            self, mock_model_class, mock_tokenizer_class):
        """Test a clear error is raised when no output length is provided."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = 'cpu'
        mock_model_class.from_pretrained.return_value = mock_model

        model = HuggingFace(
            path='test/model/path',
            max_seq_len=2048,
            model_kwargs=dict(device_map='cpu'),
        )

        with self.assertRaisesRegex(ValueError, '`max_out_len` is required'):
            model.generate(['Hello'], max_out_len=None)

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    def test_get_token_len(self, mock_model_class, mock_tokenizer_class):
        """Test get_token_len method."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model = MagicMock()
        mock_model.device = 'cpu'
        mock_model_class.from_pretrained.return_value = mock_model

        model = HuggingFace(
            path='test/model/path',
            max_seq_len=2048,
            model_kwargs=dict(device_map='cpu'),
        )

        token_len = model.get_token_len('Hello')

        self.assertEqual(token_len, 5)
        mock_tokenizer.encode.assert_called_once_with('Hello')

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    @patch('opencompass.models.huggingface.MultiTokenEOSCriteria')
    @patch('opencompass.models.huggingface.transformers')
    @patch('opencompass.models.huggingface.torch')
    def test_generate_with_mid_mode(self, mock_torch, mock_transformers,
                                    mock_stopping_criteria_class,
                                    mock_model_class, mock_tokenizer_class):
        """Test generate with mode='mid'."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token = None
        # Mock tokenizer call to return input_ids
        mock_tokenizer.return_value = {'input_ids': [[1] * 3000]}
        mock_tokenizer.decode.return_value = 'Truncated prompt'
        mock_tokenizer.batch_decode.return_value = ['Generated response']
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        # Mock torch.tensor to return a mock tensor
        mock_input_tensor = MagicMock()
        mock_input_tensor.shape = [1, 3000]  # batch_size=1, seq_len=3000
        # Support indexing: input_ids[0]
        mock_input_tensor.__getitem__.return_value = MagicMock()
        mock_input_tensor.__getitem__.return_value.__len__.return_value = 3000
        # For the second call (after truncation)
        mock_input_tensor_short = MagicMock()
        mock_input_tensor_short.shape = [1, 3]  # batch_size=1, seq_len=3
        mock_torch.tensor.side_effect = [
            mock_input_tensor, mock_input_tensor_short
        ]
        mock_model = MagicMock()
        mock_model.device = 'cpu'
        # Mock output tensor that supports slicing
        mock_output = MagicMock()
        mock_output.shape = (1, 5)
        # Support slicing: outputs[:, input_ids.shape[1]:]
        mock_output.__getitem__.return_value = mock_output
        mock_model.generate.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model

        mock_stopping_criteria = MagicMock()
        mock_stopping_criteria_class.return_value = mock_stopping_criteria
        mock_stopping_criteria_list = MagicMock()
        mock_transformers.StoppingCriteriaList.return_value = mock_stopping_criteria_list  # noqa

        model = HuggingFace(
            path='test/model/path',
            max_seq_len=2048,
            mode='mid',
            model_kwargs=dict(device_map='cpu'),
        )

        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(len(results), 1)
        mock_model.generate.assert_called()

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    @patch('opencompass.models.huggingface.MultiTokenEOSCriteria')
    @patch('opencompass.models.huggingface.transformers')
    def test_generate_with_batch_padding(self, mock_transformers,
                                         mock_stopping_criteria_class,
                                         mock_model_class,
                                         mock_tokenizer_class):
        """Test generate with batch_padding=True."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.batch_encode_plus.return_value = {
            'input_ids': [[1, 2, 3], [4, 5, 6]],
            'attention_mask': [[1, 1, 1], [1, 1, 1]]
        }
        mock_tokenizer.batch_decode.return_value = ['Response 1', 'Response 2']
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model = MagicMock()
        mock_model.device = 'cpu'
        mock_output = MagicMock()
        mock_output.shape = (2, 5)
        mock_model.generate.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model

        mock_stopping_criteria = MagicMock()
        mock_stopping_criteria_class.return_value = mock_stopping_criteria
        mock_stopping_criteria_list = MagicMock()
        mock_transformers.StoppingCriteriaList.return_value = mock_stopping_criteria_list

        model = HuggingFace(
            path='test/model/path',
            max_seq_len=2048,
            batch_padding=True,
            model_kwargs=dict(device_map='cpu'),
        )

        results = model.generate(['Hello', 'Hi'], max_out_len=100)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], 'Response 1')
        self.assertEqual(results[1], 'Response 2')
        mock_model.generate.assert_called_once()

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    def test_get_ppl(self, mock_model_class, mock_tokenizer_class):
        """Test get_ppl method."""
        import numpy as np
        import torch

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.__call__ = MagicMock(
            return_value={'input_ids': [[1, 2, 3, 4, 5]]})
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model = MagicMock()
        mock_model.device = 'cpu'
        # Mock model output logits
        mock_output = MagicMock()
        mock_output.shape = (1, 5, 1000)  # (batch, seq_len, vocab_size)
        mock_model.return_value = (mock_output, )
        mock_model_class.from_pretrained.return_value = mock_model

        model = HuggingFace(
            path='test/model/path',
            max_seq_len=2048,
            model_kwargs=dict(device_map='cpu'),
        )

        # Mock the get_logits method to return expected values
        with patch.object(model, 'get_logits') as mock_get_logits:
            mock_logits = torch.randn(1, 5, 1000)
            mock_tokens = {'input_ids': torch.tensor([[1, 2, 3, 4, 5]])}
            mock_get_logits.return_value = (mock_logits, {
                'tokens': mock_tokens
            })

            results = model.get_ppl(['Hello'])

            self.assertEqual(len(results), 1)
            self.assertIsInstance(results[0], (float, np.floating))

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    def test_pad_token_id_read_only_property_issue_725(
            self, mock_model_class, mock_tokenizer_class):
        """Regression test for issue #725.

        Some tokenizers (e.g. ChatGLM3's ``ChatGLMTokenizer``) expose
        ``pad_token_id`` as a read-only ``@property`` without a setter.
        Assigning to it previously raised
        ``AttributeError: can't set attribute 'pad_token_id'`` during
        ``_load_tokenizer``. The fix catches this and falls back to the
        tokenizer's existing ``pad_token_id`` with a warning instead of
        crashing model initialization.
        """
        mock_tokenizer = _ReadOnlyPadTokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model = MagicMock()
        mock_model.device = 'cpu'
        mock_model_class.from_pretrained.return_value = mock_model

        # Before the fix this raised:
        #   AttributeError: can't set attribute 'pad_token_id'
        model = HuggingFace(
            path='test/model/path',
            max_seq_len=2048,
            pad_token_id=151643,
            model_kwargs=dict(device_map='cpu'),
        )

        # The user-requested value is preserved on the wrapper; the
        # tokenizer's read-only value remains unchanged.
        self.assertEqual(model.pad_token_id, 151643)
        self.assertEqual(model.tokenizer.pad_token_id, 0)

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    def test_pad_token_id_negative_with_read_only_property_issue_725(
            self, mock_model_class, mock_tokenizer_class):
        """Edge case for issue #725: negative pad_token_id with a
        read-only property tokenizer.

        A negative ``pad_token_id`` is normalized via
        ``self.pad_token_id += tokenizer.vocab_size`` *before* the
        assignment. The read-only-property guard must still catch the
        subsequent ``AttributeError``.
        """
        mock_tokenizer = _ReadOnlyPadTokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model = MagicMock()
        mock_model.device = 'cpu'
        mock_model_class.from_pretrained.return_value = mock_model

        model = HuggingFace(
            path='test/model/path',
            max_seq_len=2048,
            pad_token_id=-1,
            model_kwargs=dict(device_map='cpu'),
        )

        # -1 + vocab_size(65024) = 65023, preserved on the wrapper
        self.assertEqual(model.pad_token_id, 65023)
        self.assertEqual(model.tokenizer.pad_token_id, 0)


if __name__ == '__main__':
    unittest.main()
