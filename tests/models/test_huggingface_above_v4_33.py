"""Unit tests for HuggingFacewithChatTemplate and HuggingFaceBaseModel."""

import unittest
from unittest.mock import MagicMock, patch

from opencompass.models.huggingface_above_v4_33 import (
    HuggingFaceBaseModel, HuggingFacewithChatTemplate)


class TestHuggingFacewithChatTemplate(unittest.TestCase):
    """Test cases for HuggingFacewithChatTemplate."""

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    @patch(
        'opencompass.models.huggingface_above_v4_33._get_possible_max_seq_len')
    @patch('opencompass.models.huggingface_above_v4_33._get_meta_template')
    def test_initialization_basic(self, mock_get_meta_template,
                                  mock_get_max_seq_len, mock_model_class,
                                  mock_tokenizer_class):
        """Test basic initialization."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model = MagicMock()
        mock_model.device = 'cpu'
        mock_model_class.from_pretrained.return_value = mock_model

        model = HuggingFacewithChatTemplate(
            path='test/model/path',
            model_kwargs=dict(device_map='cpu'),
            generation_kwargs=dict(temperature=0.7),
            max_seq_len=2048,
        )

        self.assertEqual(model.path, 'test/model/path')
        self.assertEqual(model.max_seq_len, 2048)
        self.assertEqual(model.generation_kwargs, dict(temperature=0.7))
        self.assertFalse(model.tokenizer_only)
        self.assertEqual(model.model, mock_model)

    @patch('transformers.AutoTokenizer')
    @patch(
        'opencompass.models.huggingface_above_v4_33._get_possible_max_seq_len')
    @patch('opencompass.models.huggingface_above_v4_33._get_meta_template')
    def test_initialization_tokenizer_only(self, mock_get_meta_template,
                                           mock_get_max_seq_len,
                                           mock_tokenizer_class):
        """Test initialization with tokenizer_only=True."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        model = HuggingFacewithChatTemplate(
            path='test/model/path',
            tokenizer_only=True,
            max_seq_len=2048,
        )

        self.assertTrue(model.tokenizer_only)
        self.assertFalse(hasattr(model, 'model'))
        mock_tokenizer_class.from_pretrained.assert_called_once()

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    @patch(
        'opencompass.models.huggingface_above_v4_33._get_possible_max_seq_len')
    @patch('opencompass.models.huggingface_above_v4_33._get_meta_template')
    @patch('opencompass.models.huggingface_above_v4_33._convert_chat_messages')
    @patch('opencompass.models.huggingface_above_v4_33._get_stopping_criteria')
    def test_generate_basic(self, mock_get_stopping_criteria,
                            mock_convert_messages, mock_get_meta_template,
                            mock_get_max_seq_len, mock_model_class,
                            mock_tokenizer_class):
        """Test basic generate functionality."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token = None
        # Mock tensors that support .to() method
        mock_input_ids = MagicMock()
        mock_input_ids.to.return_value = mock_input_ids
        mock_input_ids.shape = [1, 3]  # batch_size=1, seq_len=3
        mock_attention_mask = MagicMock()
        mock_attention_mask.to.return_value = mock_attention_mask
        mock_tokenizer.batch_encode_plus.return_value = {
            'input_ids': mock_input_ids,
            'attention_mask': mock_attention_mask
        }
        mock_tokenizer.apply_chat_template.return_value = 'Formatted prompt'
        mock_tokenizer.batch_decode.return_value = ['Generated response']
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model = MagicMock()
        mock_model.device = 'cpu'
        # Mock output tensor that supports slicing
        mock_output = MagicMock()
        mock_output.shape = (1, 5)
        # Support slicing: outputs[:, tokens['input_ids'].shape[1]:]
        mock_output.__getitem__.return_value = mock_output
        mock_model.generate.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model

        mock_convert_messages.return_value = [{
            'role': 'user',
            'content': 'Hello'
        }]
        mock_stopping_criteria = MagicMock()
        mock_get_stopping_criteria.return_value = mock_stopping_criteria

        model = HuggingFacewithChatTemplate(
            path='test/model/path',
            model_kwargs=dict(device_map='cpu'),
            generation_kwargs=dict(temperature=0.7),
            max_seq_len=2048,
        )

        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], 'Generated response')
        mock_model.generate.assert_called_once()

    @patch('transformers.AutoTokenizer')
    @patch(
        'opencompass.models.huggingface_above_v4_33._get_possible_max_seq_len')
    @patch('opencompass.models.huggingface_above_v4_33._get_meta_template')
    @patch('opencompass.models.huggingface_above_v4_33._convert_chat_messages')
    def test_get_token_len(self, mock_convert_messages, mock_get_meta_template,
                           mock_get_max_seq_len, mock_tokenizer_class):
        """Test get_token_len method."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token = None
        mock_tokenizer.apply_chat_template.return_value = {
            'input_ids': [1, 2, 3, 4, 5]
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_convert_messages.return_value = [{
            'role': 'user',
            'content': 'Hello'
        }]

        model = HuggingFacewithChatTemplate(
            path='test/model/path',
            tokenizer_only=True,
            max_seq_len=2048,
        )

        token_len = model.get_token_len('Hello')

        self.assertEqual(token_len, 5)
        mock_convert_messages.assert_called_once_with(['Hello'])
        mock_tokenizer.apply_chat_template.assert_called_once()


class TestHuggingFaceBaseModel(unittest.TestCase):
    """Test cases for HuggingFaceBaseModel."""

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    @patch(
        'opencompass.models.huggingface_above_v4_33._get_possible_max_seq_len')
    def test_initialization_basic(self, mock_get_max_seq_len, mock_model_class,
                                  mock_tokenizer_class):
        """Test basic initialization."""
        mock_get_max_seq_len.return_value = 2048
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model = MagicMock()
        mock_model.device = 'cpu'
        mock_model_class.from_pretrained.return_value = mock_model

        model = HuggingFaceBaseModel(
            path='test/model/path',
            model_kwargs=dict(device_map='cpu'),
            generation_kwargs=dict(temperature=0.7),
            max_seq_len=2048,
        )

        self.assertEqual(model.path, 'test/model/path')
        self.assertEqual(model.max_seq_len, 2048)
        self.assertEqual(model.generation_kwargs, dict(temperature=0.7))
        self.assertFalse(model.tokenizer_only)
        self.assertFalse(model.drop_middle)

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    @patch(
        'opencompass.models.huggingface_above_v4_33._get_possible_max_seq_len')
    @patch('opencompass.models.huggingface_above_v4_33._convert_base_messages')
    @patch('opencompass.models.huggingface_above_v4_33._get_stopping_criteria')
    def test_generate_basic(self, mock_get_stopping_criteria,
                            mock_convert_base_messages, mock_get_max_seq_len,
                            mock_model_class, mock_tokenizer_class):
        """Test basic generate functionality."""
        mock_get_max_seq_len.return_value = 2048
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token = None
        # Mock tensors that support .to() method
        mock_input_ids = MagicMock()
        mock_input_ids.to.return_value = mock_input_ids
        mock_attention_mask = MagicMock()
        mock_attention_mask.to.return_value = mock_attention_mask
        mock_tokenizer.batch_encode_plus.return_value = {
            'input_ids': mock_input_ids,
            'attention_mask': mock_attention_mask
        }
        mock_tokenizer.batch_decode.return_value = ['Generated response']
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model = MagicMock()
        mock_model.device = 'cpu'
        mock_output = MagicMock()
        mock_output.shape = (1, 5)
        mock_model.generate.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model

        mock_convert_base_messages.return_value = ['Hello']
        mock_stopping_criteria = MagicMock()
        mock_get_stopping_criteria.return_value = mock_stopping_criteria

        model = HuggingFaceBaseModel(
            path='test/model/path',
            model_kwargs=dict(device_map='cpu'),
            generation_kwargs=dict(temperature=0.7),
            max_seq_len=2048,
        )

        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], 'Generated response')
        mock_model.generate.assert_called_once()

    @patch('transformers.AutoTokenizer')
    @patch(
        'opencompass.models.huggingface_above_v4_33._get_possible_max_seq_len')
    @patch('opencompass.models.huggingface_above_v4_33._convert_base_messages')
    def test_get_token_len(self, mock_convert_base_messages,
                           mock_get_max_seq_len, mock_tokenizer_class):
        """Test get_token_len method."""
        mock_get_max_seq_len.return_value = 2048
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token = None
        mock_tokenizer.return_value = {'input_ids': [1, 2, 3, 4, 5]}
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_convert_base_messages.return_value = ['Hello']

        model = HuggingFaceBaseModel(
            path='test/model/path',
            tokenizer_only=True,
            max_seq_len=2048,
        )

        token_len = model.get_token_len('Hello', add_special_tokens=True)

        self.assertEqual(token_len, 5)
        mock_convert_base_messages.assert_called_once_with(['Hello'])
        mock_tokenizer.assert_called_once()


if __name__ == '__main__':
    unittest.main()
