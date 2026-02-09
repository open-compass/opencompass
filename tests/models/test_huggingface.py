# docformatter: noqa
"""Unit tests for HuggingFace."""

import unittest
from unittest.mock import MagicMock, patch

from opencompass.models.huggingface import HuggingFace


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


if __name__ == '__main__':
    unittest.main()
