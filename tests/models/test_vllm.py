"""Unit tests for VLLM."""

import unittest
from unittest.mock import MagicMock, patch

from opencompass.models.vllm import VLLM


class TestVLLM(unittest.TestCase):
    """Test cases for VLLM."""

    @patch('opencompass.models.vllm.LLM')
    @patch('opencompass.models.vllm.ray', create=True)
    def test_initialization_basic(self, mock_ray, mock_llm_class):
        """Test basic initialization."""
        mock_ray.is_initialized.return_value = False
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.get_tokenizer.return_value = mock_tokenizer
        mock_llm_class.return_value = mock_model

        model = VLLM(
            path='test/model/path',
            max_seq_len=2048,
            model_kwargs=dict(tensor_parallel_size=1),
            generation_kwargs=dict(temperature=0.7),
        )

        self.assertEqual(model.path, 'test/model/path')
        self.assertEqual(model.max_seq_len, 2048)
        self.assertEqual(model.generation_kwargs, dict(temperature=0.7))
        self.assertEqual(model.model, mock_model)
        self.assertEqual(model.mode, 'none')

    @patch('opencompass.models.vllm.LLM')
    @patch('opencompass.models.vllm.ray', create=True)
    @patch('opencompass.models.vllm.SamplingParams')
    def test_generate_basic(self, mock_sampling_params_class, mock_ray,
                            mock_llm_class):
        """Test basic generate functionality."""
        mock_ray.is_initialized.return_value = False
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.get_tokenizer.return_value = mock_tokenizer
        mock_llm_class.return_value = mock_model

        mock_output = MagicMock()
        mock_output.prompt = 'Hello'
        mock_output.outputs = [MagicMock(text='Generated response')]
        mock_model.generate.return_value = [mock_output]
        mock_sampling_params = MagicMock()
        mock_sampling_params_class.return_value = mock_sampling_params

        model = VLLM(
            path='test/model/path',
            max_seq_len=2048,
            model_kwargs=dict(tensor_parallel_size=1),
            generation_kwargs=dict(temperature=0.7),
        )

        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], 'Generated response')
        mock_model.generate.assert_called_once()

    @patch('opencompass.models.vllm.LLM')
    @patch('opencompass.models.vllm.ray', create=True)
    @patch('opencompass.models.vllm.SamplingParams')
    def test_generate_with_mid_mode(self, mock_sampling_params_class, mock_ray,
                                    mock_llm_class):
        """Test generate with mode='mid'."""
        mock_ray.is_initialized.return_value = False
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.__call__ = MagicMock(
            return_value={'input_ids': [[1] * 3000]})
        mock_tokenizer.decode.return_value = 'Truncated prompt'
        mock_model.get_tokenizer.return_value = mock_tokenizer
        mock_llm_class.return_value = mock_model

        mock_output = MagicMock()
        mock_output.prompt = 'Hello'
        mock_output.outputs = [MagicMock(text='Generated response')]
        mock_model.generate.return_value = [mock_output]
        mock_sampling_params = MagicMock()
        mock_sampling_params_class.return_value = mock_sampling_params

        model = VLLM(
            path='test/model/path',
            max_seq_len=2048,
            mode='mid',
            model_kwargs=dict(tensor_parallel_size=1),
        )

        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(len(results), 1)
        mock_model.generate.assert_called_once()

    @patch('opencompass.models.vllm.LLM')
    @patch('opencompass.models.vllm.ray', create=True)
    def test_get_token_len(self, mock_ray, mock_llm_class):
        """Test get_token_len method."""
        mock_ray.is_initialized.return_value = False
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_model.get_tokenizer.return_value = mock_tokenizer
        mock_llm_class.return_value = mock_model

        model = VLLM(
            path='test/model/path',
            max_seq_len=2048,
            model_kwargs=dict(tensor_parallel_size=1),
        )

        token_len = model.get_token_len('Hello', add_special_tokens=True)

        self.assertEqual(token_len, 5)
        mock_tokenizer.encode.assert_called_once_with('Hello',
                                                      add_special_tokens=True)

    @patch('opencompass.models.vllm.LLM')
    @patch('opencompass.models.vllm.ray', create=True)
    @patch('opencompass.models.vllm.SamplingParams')
    @patch('opencompass.models.vllm.LoRARequest', create=True)
    def test_generate_with_lora(self, mock_lora_request_class,
                                mock_sampling_params_class, mock_ray,
                                mock_llm_class):
        """Test generate with LoRA adapter."""
        mock_ray.is_initialized.return_value = False
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.get_tokenizer.return_value = mock_tokenizer
        mock_llm_class.return_value = mock_model

        mock_output = MagicMock()
        mock_output.prompt = 'Hello'
        mock_output.outputs = [MagicMock(text='Generated response')]
        mock_model.generate.return_value = [mock_output]
        mock_sampling_params = MagicMock()
        mock_sampling_params_class.return_value = mock_sampling_params
        mock_lora_request = MagicMock()
        mock_lora_request_class.return_value = mock_lora_request

        model = VLLM(
            path='test/model/path',
            max_seq_len=2048,
            model_kwargs=dict(tensor_parallel_size=1),
            lora_path='test/lora/path',
        )

        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(len(results), 1)
        # Verify LoRA request is used
        mock_model.generate.assert_called_once()
        call_args = mock_model.generate.call_args
        self.assertIn('lora_request', call_args.kwargs)
        self.assertEqual(call_args.kwargs['lora_request'], mock_lora_request)

    @patch('opencompass.models.vllm.LLM')
    @patch('opencompass.models.vllm.ray', create=True)
    @patch('opencompass.models.vllm.SamplingParams')
    def test_generate_with_stop_words(self, mock_sampling_params_class,
                                      mock_ray, mock_llm_class):
        """Test generate with stop_words."""
        mock_ray.is_initialized.return_value = False
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.get_tokenizer.return_value = mock_tokenizer
        mock_llm_class.return_value = mock_model

        mock_output = MagicMock()
        mock_output.prompt = 'Hello'
        mock_output.outputs = [MagicMock(text='Generated response')]
        mock_model.generate.return_value = [mock_output]
        mock_sampling_params = MagicMock()
        mock_sampling_params_class.return_value = mock_sampling_params

        model = VLLM(
            path='test/model/path',
            max_seq_len=2048,
            model_kwargs=dict(tensor_parallel_size=1),
            stop_words=['stop_word'],
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


if __name__ == '__main__':
    unittest.main()
