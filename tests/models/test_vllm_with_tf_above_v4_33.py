"""Unit tests for VLLMwithChatTemplate."""

import unittest
from unittest.mock import MagicMock, patch

from opencompass.models.vllm_with_tf_above_v4_33 import VLLMwithChatTemplate


class TestVLLMwithChatTemplate(unittest.TestCase):
    """Test cases for VLLMwithChatTemplate."""

    @patch('opencompass.models.vllm_with_tf_above_v4_33.LLM')
    @patch('transformers.AutoTokenizer')
    @patch(
        'opencompass.models.vllm_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch('opencompass.models.vllm_with_tf_above_v4_33._get_meta_template')
    @patch('opencompass.models.vllm_with_tf_above_v4_33.ray', create=True)
    def test_initialization_basic(self, mock_ray, mock_get_meta_template,
                                  mock_get_max_seq_len, mock_tokenizer_class,
                                  mock_llm_class):
        """Test basic initialization."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_ray.is_initialized.return_value = False
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = None
        mock_tokenizer.eos_token = None
        mock_model.get_tokenizer.return_value = mock_tokenizer
        mock_llm_class.return_value = mock_model

        model = VLLMwithChatTemplate(
            path='test/model/path',
            model_kwargs=dict(tensor_parallel_size=1),
            generation_kwargs=dict(temperature=0.7),
            max_seq_len=2048,
        )

        self.assertEqual(model.path, 'test/model/path')
        self.assertEqual(model.max_seq_len, 2048)
        self.assertEqual(model.generation_kwargs, dict(temperature=0.7))
        self.assertFalse(model.tokenizer_only)
        self.assertEqual(model.model, mock_model)

    @patch('opencompass.models.vllm_with_tf_above_v4_33.LLM')
    @patch('transformers.AutoTokenizer')
    @patch(
        'opencompass.models.vllm_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch('opencompass.models.vllm_with_tf_above_v4_33._get_meta_template')
    def test_initialization_tokenizer_only(self, mock_get_meta_template,
                                           mock_get_max_seq_len,
                                           mock_tokenizer_class,
                                           mock_llm_class):
        """Test initialization with tokenizer_only=True."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        model = VLLMwithChatTemplate(
            path='test/model/path',
            tokenizer_only=True,
            max_seq_len=2048,
        )

        self.assertTrue(model.tokenizer_only)
        self.assertFalse(hasattr(model, 'model'))
        mock_tokenizer_class.from_pretrained.assert_called_once()

    @patch('opencompass.models.vllm_with_tf_above_v4_33.LLM')
    @patch('transformers.AutoTokenizer')
    @patch(
        'opencompass.models.vllm_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch('opencompass.models.vllm_with_tf_above_v4_33._get_meta_template')
    @patch('opencompass.models.vllm_with_tf_above_v4_33._convert_chat_messages'
           )
    @patch('opencompass.models.vllm_with_tf_above_v4_33.SamplingParams')
    @patch('opencompass.models.vllm_with_tf_above_v4_33.ray', create=True)
    def test_generate_basic(self, mock_ray, mock_sampling_params_class,
                            mock_convert_messages, mock_get_meta_template,
                            mock_get_max_seq_len, mock_tokenizer_class,
                            mock_llm_class):
        """Test basic generate functionality."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_ray.is_initialized.return_value = False
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = None
        mock_tokenizer.eos_token = None
        mock_model.get_tokenizer.return_value = mock_tokenizer
        mock_llm_class.return_value = mock_model

        mock_convert_messages.return_value = [{
            'role': 'user',
            'content': 'Hello'
        }]
        mock_tokenizer.apply_chat_template.return_value = 'Formatted prompt'
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text='Generated response')]
        mock_model.generate.return_value = [mock_output]
        mock_sampling_params = MagicMock()
        mock_sampling_params_class.return_value = mock_sampling_params

        model = VLLMwithChatTemplate(
            path='test/model/path',
            model_kwargs=dict(tensor_parallel_size=1),
            generation_kwargs=dict(temperature=0.7),
            max_seq_len=2048,
        )

        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], 'Generated response')
        mock_model.generate.assert_called_once()

    @patch('opencompass.models.vllm_with_tf_above_v4_33.LLM')
    @patch('transformers.AutoTokenizer')
    @patch(
        'opencompass.models.vllm_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch('opencompass.models.vllm_with_tf_above_v4_33._get_meta_template')
    @patch('opencompass.models.vllm_with_tf_above_v4_33._convert_chat_messages'
           )
    def test_get_token_len(self, mock_convert_messages, mock_get_meta_template,
                           mock_get_max_seq_len, mock_tokenizer_class,
                           mock_llm_class):
        """Test get_token_len method."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = False
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.get_tokenizer.return_value = mock_tokenizer
        mock_llm_class.return_value = mock_model

        mock_convert_messages.return_value = [{
            'role': 'user',
            'content': 'Hello'
        }]
        mock_tokenizer.apply_chat_template.return_value = {
            'input_ids': [1, 2, 3, 4, 5]
        }

        with patch('opencompass.models.vllm_with_tf_above_v4_33.ray',
                   create=True) as mock_ray_patch:
            mock_ray_patch.is_initialized.return_value = False
            model = VLLMwithChatTemplate(
                path='test/model/path',
                model_kwargs=dict(tensor_parallel_size=1),
                max_seq_len=2048,
            )

            token_len = model.get_token_len('Hello')

            self.assertEqual(token_len, 5)
            mock_convert_messages.assert_called_once_with(['Hello'])
            mock_tokenizer.apply_chat_template.assert_called_once()

    @patch('opencompass.models.vllm_with_tf_above_v4_33.LLM')
    @patch('transformers.AutoTokenizer')
    @patch(
        'opencompass.models.vllm_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch('opencompass.models.vllm_with_tf_above_v4_33._get_meta_template')
    @patch('opencompass.models.vllm_with_tf_above_v4_33._convert_chat_messages'
           )
    @patch('opencompass.models.vllm_with_tf_above_v4_33.SamplingParams')
    @patch('opencompass.models.vllm_with_tf_above_v4_33.ray', create=True)
    def test_generate_with_stop_words(self, mock_ray,
                                      mock_sampling_params_class,
                                      mock_convert_messages,
                                      mock_get_meta_template,
                                      mock_get_max_seq_len,
                                      mock_tokenizer_class, mock_llm_class):
        """Test generate with stop_words."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_ray.is_initialized.return_value = False
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = None
        mock_tokenizer.eos_token = None
        mock_model.get_tokenizer.return_value = mock_tokenizer
        mock_llm_class.return_value = mock_model

        mock_convert_messages.return_value = [{
            'role': 'user',
            'content': 'Hello'
        }]
        mock_tokenizer.apply_chat_template.return_value = 'Formatted prompt'
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text='Generated response')]
        mock_model.generate.return_value = [mock_output]
        mock_sampling_params = MagicMock()
        mock_sampling_params_class.return_value = mock_sampling_params

        model = VLLMwithChatTemplate(
            path='test/model/path',
            model_kwargs=dict(tensor_parallel_size=1),
            stop_words=['stop_word'],
            max_seq_len=2048,
        )

        results = model.generate(['Hello'],
                                 max_out_len=100,
                                 stopping_criteria=['extra_stop'])

        self.assertEqual(len(results), 1)
        # Verify stop words are included in sampling params
        call_args = mock_sampling_params_class.call_args[1]
        self.assertIn('stop', call_args)
        self.assertIn('stop_word', call_args['stop'])
        self.assertIn('extra_stop', call_args['stop'])

    @patch('opencompass.models.vllm_with_tf_above_v4_33.LLM')
    @patch('transformers.AutoTokenizer')
    @patch(
        'opencompass.models.vllm_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch('opencompass.models.vllm_with_tf_above_v4_33._get_meta_template')
    @patch('opencompass.models.vllm_with_tf_above_v4_33._convert_chat_messages'
           )
    @patch('opencompass.models.vllm_with_tf_above_v4_33.SamplingParams')
    @patch('opencompass.models.vllm_with_tf_above_v4_33.ray', create=True)
    @patch('vllm.lora.request.LoRARequest')
    def test_generate_with_lora(self, mock_lora_request_class, mock_ray,
                                mock_sampling_params_class,
                                mock_convert_messages, mock_get_meta_template,
                                mock_get_max_seq_len, mock_tokenizer_class,
                                mock_llm_class):
        """Test generate with LoRA adapter."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_ray.is_initialized.return_value = False
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = None
        mock_model.get_tokenizer.return_value = mock_tokenizer
        mock_llm_class.return_value = mock_model

        mock_convert_messages.return_value = [{
            'role': 'user',
            'content': 'Hello'
        }]
        mock_tokenizer.apply_chat_template.return_value = 'Formatted prompt'
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text='Generated response')]
        mock_model.generate.return_value = [mock_output]
        mock_sampling_params = MagicMock()
        mock_sampling_params_class.return_value = mock_sampling_params
        mock_lora_request = MagicMock()
        mock_lora_request_class.return_value = mock_lora_request

        model = VLLMwithChatTemplate(
            path='test/model/path',
            model_kwargs=dict(tensor_parallel_size=1),
            lora_path='test/lora/path',
            max_seq_len=2048,
        )

        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(len(results), 1)
        # Verify LoRA request is used
        mock_model.generate.assert_called_once()
        call_args = mock_model.generate.call_args
        self.assertIn('lora_request', call_args.kwargs)
        self.assertEqual(call_args.kwargs['lora_request'], mock_lora_request)


if __name__ == '__main__':
    unittest.main()
