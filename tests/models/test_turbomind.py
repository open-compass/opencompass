"""Unit tests for TurboMindModel."""

import unittest
from unittest.mock import MagicMock, patch

from opencompass.models.turbomind import TurboMindModel


class TestTurboMindModel(unittest.TestCase):
    """Test cases for TurboMindModel."""

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch('opencompass.models.turbomind._get_possible_max_seq_len')
    def test_initialization_basic(self, mock_get_max_seq_len, mock_pipeline,
                                  mock_tokenizer_class):
        """Test basic initialization."""
        mock_get_max_seq_len.return_value = 2048
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe

        model = TurboMindModel(
            path='test/model/path',
            backend='turbomind',
            max_seq_len=2048,
            engine_config=dict(session_len=4096, max_batch_size=1),
            gen_config=dict(do_sample=False),
        )

        self.assertEqual(model.path, 'test/model/path')
        self.assertEqual(model.max_seq_len, 2048)
        self.assertEqual(model.gen_config, dict(do_sample=False))
        self.assertEqual(model.pipe, mock_pipe)
        self.assertFalse(model.drop_middle)
        self.assertFalse(model.batch_padding)

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch('opencompass.models.turbomind._get_possible_max_seq_len')
    def test_initialization_with_drop_middle(self, mock_get_max_seq_len,
                                             mock_pipeline,
                                             mock_tokenizer_class):
        """Test initialization with drop_middle=True."""
        mock_get_max_seq_len.return_value = 2048
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe

        model = TurboMindModel(
            path='test/model/path',
            drop_middle=True,
            max_seq_len=2048,
        )

        self.assertTrue(model.drop_middle)

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch('opencompass.models.turbomind._get_possible_max_seq_len')
    @patch('lmdeploy.GenerationConfig')
    def test_generate_basic(self, mock_gen_config_class, mock_get_max_seq_len,
                            mock_pipeline, mock_tokenizer_class):
        """Test basic generate functionality."""
        mock_get_max_seq_len.return_value = 2048
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.token_ids = [1, 2, 3, 4, 5]
        mock_pipe.return_value = [mock_output]
        mock_pipeline.return_value = mock_pipe

        mock_tokenizer.decode.return_value = 'Generated response'
        mock_gen_config = MagicMock()
        mock_gen_config_class.return_value = mock_gen_config

        model = TurboMindModel(
            path='test/model/path',
            engine_config=dict(session_len=4096, max_batch_size=1),
            gen_config=dict(do_sample=False),
            max_seq_len=2048,
        )

        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(len(results), 1)
        # When stop_words is empty, no split is performed, so result is unchanged
        self.assertEqual(results[0], 'Generated response')
        mock_pipe.assert_called_once()
        mock_tokenizer.decode.assert_called()

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch('opencompass.models.turbomind._get_possible_max_seq_len')
    @patch('lmdeploy.GenerationConfig')
    def test_generate_with_stop_words(self, mock_gen_config_class,
                                      mock_get_max_seq_len, mock_pipeline,
                                      mock_tokenizer_class):
        """Test generate with stop_words."""
        mock_get_max_seq_len.return_value = 2048
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.token_ids = [1, 2, 3, 4, 5]
        mock_pipe.return_value = [mock_output]
        mock_pipeline.return_value = mock_pipe

        mock_tokenizer.decode.return_value = 'Generated response stop_word more text'
        mock_gen_config = MagicMock()
        mock_gen_config_class.return_value = mock_gen_config

        model = TurboMindModel(
            path='test/model/path',
            engine_config=dict(session_len=4096, max_batch_size=1),
            gen_config=dict(do_sample=False),
            max_seq_len=2048,
        )

        results = model.generate(['Hello'],
                                 max_out_len=100,
                                 stopping_criteria=['stop_word'])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], 'Generated response ')
        mock_pipe.assert_called_once()

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch('opencompass.models.turbomind._get_possible_max_seq_len')
    @patch('lmdeploy.GenerationConfig')
    def test_generate_with_drop_middle(self, mock_gen_config_class,
                                       mock_get_max_seq_len, mock_pipeline,
                                       mock_tokenizer_class):
        """Test generate with drop_middle=True."""
        mock_get_max_seq_len.return_value = 2048
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.token_ids = [1, 2, 3, 4, 5]
        mock_pipe.return_value = [mock_output]
        mock_pipeline.return_value = mock_pipe

        def tokenize_side_effect(input_list, **kwargs):
            return {'input_ids': [[1] * 3000]}

        mock_tokenizer.__call__ = MagicMock(side_effect=tokenize_side_effect)
        mock_tokenizer.decode.return_value = 'Truncated prompt'
        mock_gen_config = MagicMock()
        mock_gen_config_class.return_value = mock_gen_config

        model = TurboMindModel(
            path='test/model/path',
            engine_config=dict(session_len=4096, max_batch_size=1),
            gen_config=dict(do_sample=False),
            drop_middle=True,
            max_seq_len=2048,
        )

        results = model.generate(['Hello'], max_out_len=100)

        self.assertEqual(len(results), 1)
        mock_pipe.assert_called_once()

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch('opencompass.models.turbomind._get_possible_max_seq_len')
    def test_get_token_len(self, mock_get_max_seq_len, mock_pipeline,
                           mock_tokenizer_class):
        """Test get_token_len method."""
        mock_get_max_seq_len.return_value = 2048
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe

        model = TurboMindModel(
            path='test/model/path',
            engine_config=dict(session_len=4096, max_batch_size=1),
            max_seq_len=2048,
        )

        token_len = model.get_token_len('Hello')

        self.assertEqual(token_len, 5)
        mock_tokenizer.encode.assert_called_once_with('Hello')

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 5, 0))
    @patch('lmdeploy.pipeline')
    @patch('opencompass.models.turbomind._get_possible_max_seq_len')
    @patch('lmdeploy.GenerationConfig')
    def test_generate_with_old_version(self, mock_gen_config_class,
                                       mock_get_max_seq_len, mock_pipeline,
                                       mock_tokenizer_class):
        """Test generate with old lmdeploy version (< 0.6.0)."""
        mock_get_max_seq_len.return_value = 2048
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.token_ids = [1, 2, 3, 4, 5]
        mock_pipe.return_value = [mock_output]
        mock_pipeline.return_value = mock_pipe

        mock_tokenizer.decode.return_value = 'Generated response'
        mock_gen_config = MagicMock()
        mock_gen_config_class.return_value = mock_gen_config

        model = TurboMindModel(
            path='test/model/path',
            engine_config=dict(session_len=4096, max_batch_size=1),
            gen_config=dict(do_sample=False),
            max_seq_len=2048,
        )

        model.generate(['Hello'], max_out_len=100, do_sample=False)

        # Verify that top_k=1 is set for old version
        call_args = mock_gen_config_class.call_args[1]
        self.assertIn('top_k', call_args)
        self.assertEqual(call_args['top_k'], 1)

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch('opencompass.models.turbomind._get_possible_max_seq_len')
    @patch('lmdeploy.GenerationConfig')
    def test_generate_with_do_sample(self, mock_gen_config_class,
                                     mock_get_max_seq_len, mock_pipeline,
                                     mock_tokenizer_class):
        """Test generate with do_sample=True."""
        mock_get_max_seq_len.return_value = 2048
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.token_ids = [1, 2, 3, 4, 5]
        mock_pipe.return_value = [mock_output]
        mock_pipeline.return_value = mock_pipe

        mock_tokenizer.decode.return_value = 'Generated response'
        mock_gen_config = MagicMock()
        mock_gen_config_class.return_value = mock_gen_config

        model = TurboMindModel(
            path='test/model/path',
            engine_config=dict(session_len=4096, max_batch_size=1),
            gen_config=dict(do_sample=False),
            max_seq_len=2048,
        )

        _ = model.generate(['Hello'],
                           max_out_len=100,
                           do_sample=True,
                           temperature=0.7)

        # Verify that top_k and temperature are set
        call_args = mock_gen_config_class.call_args[1]
        self.assertIn('top_k', call_args)
        self.assertEqual(call_args['top_k'], 40)
        self.assertIn('temperature', call_args)
        self.assertEqual(call_args['temperature'], 0.7)

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch('opencompass.models.turbomind._get_possible_max_seq_len')
    @patch('lmdeploy.TurbomindEngineConfig')
    @patch('lmdeploy.PytorchEngineConfig')
    def test_build_pipe_turbomind_backend(self, mock_pytorch_config,
                                          mock_turbomind_config,
                                          mock_get_max_seq_len, mock_pipeline,
                                          mock_tokenizer_class):
        """Test _build_pipe with turbomind backend."""
        mock_get_max_seq_len.return_value = 2048
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_turbomind_instance = MagicMock()
        mock_turbomind_config.return_value = mock_turbomind_instance
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe

        TurboMindModel(
            path='test/model/path',
            engine_config=dict(session_len=4096, max_batch_size=1),
            backend='turbomind',
            max_seq_len=2048,
        )

        mock_turbomind_config.assert_called_once()
        mock_pipeline.assert_called_once()

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch('opencompass.models.turbomind._get_possible_max_seq_len')
    @patch('lmdeploy.TurbomindEngineConfig')
    @patch('lmdeploy.PytorchEngineConfig')
    def test_build_pipe_pytorch_backend(self, mock_pytorch_config,
                                        mock_turbomind_config,
                                        mock_get_max_seq_len, mock_pipeline,
                                        mock_tokenizer_class):
        """Test _build_pipe with pytorch backend."""
        mock_get_max_seq_len.return_value = 2048
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_pytorch_instance = MagicMock()
        mock_pytorch_config.return_value = mock_pytorch_instance
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe

        TurboMindModel(
            path='test/model/path',
            engine_config=dict(session_len=4096, max_batch_size=1),
            backend='pytorch',
            max_seq_len=2048,
        )

        mock_pytorch_config.assert_called_once()
        mock_pipeline.assert_called_once()

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch('opencompass.models.turbomind._get_possible_max_seq_len')
    def test_build_pipe_invalid_backend(self, mock_get_max_seq_len,
                                        mock_pipeline, mock_tokenizer_class):
        """Test _build_pipe with invalid backend."""
        mock_get_max_seq_len.return_value = 2048
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with self.assertRaises(AssertionError):
            TurboMindModel(
                path='test/model/path',
                engine_config=dict(session_len=4096, max_batch_size=1),
                backend='invalid_backend',
                max_seq_len=2048,
            )


if __name__ == '__main__':
    unittest.main()
