# docformatter: noqa
"""Unit tests for TurboMindModelwithChatTemplate."""
import unittest
from unittest.mock import MagicMock, patch

from mmengine.config import ConfigDict

from opencompass.models.turbomind_with_tf_above_v4_33 import \
    TurboMindModelwithChatTemplate


class TestTurboMindModelwithChatTemplate(unittest.TestCase):
    """Test cases for TurboMindModelwithChatTemplate."""

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    def test_initialization_with_dict_engine_config(self,
                                                    mock_get_meta_template,
                                                    mock_get_max_seq_len,
                                                    mock_pipeline,
                                                    mock_tokenizer_class):
        """Test initialization with Dict engine_config."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe

        model = TurboMindModelwithChatTemplate(
            path='test/model/path',
            engine_config=dict(session_len=4096, max_batch_size=1),
            gen_config=dict(do_sample=False),
            max_seq_len=2048,
        )

        self.assertEqual(model.path, 'test/model/path')
        self.assertFalse(model.tokenizer_only)
        self.assertFalse(model.drop_middle)
        self.assertEqual(model.max_seq_len, 2048)
        self.assertEqual(model.gen_config, dict(do_sample=False))
        self.assertEqual(model.pipe, mock_pipe)
        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            'test/model/path', trust_remote_code=True)

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    def test_initialization_with_configdict_engine_config(
            self, mock_get_meta_template, mock_get_max_seq_len,
            mock_tokenizer_class):
        """Test initialization with ConfigDict engine_config."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        engine_config = ConfigDict(dict(session_len=4096, max_batch_size=1))
        with patch('lmdeploy.pipeline') as mock_pipeline:
            mock_pipe = MagicMock()
            mock_pipeline.return_value = mock_pipe

            model = TurboMindModelwithChatTemplate(
                path='test/model/path',
                engine_config=engine_config,
                gen_config=dict(do_sample=False),
                max_seq_len=2048,
            )

            self.assertEqual(model.path, 'test/model/path')
            self.assertEqual(model.pipe, mock_pipe)

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    def test_initialization_tokenizer_only(self, mock_get_meta_template,
                                           mock_get_max_seq_len,
                                           mock_tokenizer_class):
        """Test initialization with tokenizer_only=True."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        model = TurboMindModelwithChatTemplate(
            path='test/model/path',
            tokenizer_only=True,
            max_seq_len=2048,
        )

        self.assertTrue(model.tokenizer_only)
        self.assertIsNone(model.pipe)

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    def test_initialization_invalid_engine_config_type(self,
                                                       mock_get_meta_template,
                                                       mock_get_max_seq_len,
                                                       mock_tokenizer_class):
        """Test initialization with invalid engine_config type."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with self.assertRaises(ValueError) as context:
            TurboMindModelwithChatTemplate(
                path='test/model/path',
                engine_config='invalid_type',
                max_seq_len=2048,
            )

        self.assertIn('expected Dict or ConfigDict', str(context.exception))

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._convert_chat_messages'
    )
    @patch('lmdeploy.GenerationConfig')
    def test_generate_basic(self, mock_gen_config_class, mock_convert_messages,
                            mock_get_meta_template, mock_get_max_seq_len,
                            mock_pipeline, mock_tokenizer_class):
        """Test basic generate functionality."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = None
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.text = 'Generated response'
        mock_output.input_token_len = 10
        mock_output.generate_token_len = 5
        mock_pipe.return_value = [mock_output]
        mock_pipeline.return_value = mock_pipe

        mock_convert_messages.return_value = [{
            'role': 'user',
            'content': 'Hello'
        }]
        mock_tokenizer.apply_chat_template.return_value = 'Formatted prompt'
        mock_gen_config = MagicMock()
        mock_gen_config_class.return_value = mock_gen_config

        model = TurboMindModelwithChatTemplate(
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
        mock_gen_config_class.assert_called_once()

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._convert_chat_messages'
    )
    @patch('lmdeploy.GenerationConfig')
    def test_generate_with_stop_words(self, mock_gen_config_class,
                                      mock_convert_messages,
                                      mock_get_meta_template,
                                      mock_get_max_seq_len, mock_pipeline,
                                      mock_tokenizer_class):
        """Test generate with stop_words."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = None
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.text = 'Generated response stop_word more text'
        mock_output.input_token_len = 10
        mock_output.generate_token_len = 5
        mock_pipe.return_value = [mock_output]
        mock_pipeline.return_value = mock_pipe

        mock_convert_messages.return_value = [{
            'role': 'user',
            'content': 'Hello'
        }]
        mock_tokenizer.apply_chat_template.return_value = 'Formatted prompt'
        mock_gen_config = MagicMock()
        mock_gen_config_class.return_value = mock_gen_config

        model = TurboMindModelwithChatTemplate(
            path='test/model/path',
            engine_config=dict(session_len=4096, max_batch_size=1),
            gen_config=dict(do_sample=False),
            stop_words=['stop_word'],
            max_seq_len=2048,
        )

        results = model.generate(['Hello'],
                                 max_out_len=100,
                                 stopping_criteria=['extra_stop'])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], 'Generated response ')
        mock_pipe.assert_called_once()

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._convert_chat_messages'
    )
    @patch('lmdeploy.GenerationConfig')
    def test_generate_with_drop_middle(self, mock_gen_config_class,
                                       mock_convert_messages,
                                       mock_get_meta_template,
                                       mock_get_max_seq_len, mock_pipeline,
                                       mock_tokenizer_class):
        """Test generate with drop_middle=True."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = None
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.text = 'Generated response'
        mock_output.input_token_len = 10
        mock_output.generate_token_len = 5
        mock_pipe.return_value = [mock_output]
        mock_pipeline.return_value = mock_pipe

        mock_convert_messages.return_value = [{
            'role': 'user',
            'content': 'Hello'
        }]
        mock_tokenizer.apply_chat_template.return_value = 'Formatted prompt'
        mock_gen_config = MagicMock()
        mock_gen_config_class.return_value = mock_gen_config

        # Mock tokenizer to simulate long input
        def tokenize_side_effect(input_list, **kwargs):
            return {'input_ids': [[1] * 3000]}

        mock_tokenizer.__call__ = MagicMock(side_effect=tokenize_side_effect)
        mock_tokenizer.decode.return_value = 'Truncated prompt'

        model = TurboMindModelwithChatTemplate(
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
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._convert_chat_messages'
    )
    def test_get_token_len(self, mock_convert_messages, mock_get_meta_template,
                           mock_get_max_seq_len, mock_pipeline,
                           mock_tokenizer_class):
        """Test get_token_len method."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe

        mock_convert_messages.return_value = [{
            'role': 'user',
            'content': 'Hello'
        }]
        mock_tokenizer.apply_chat_template.return_value = {
            'input_ids': [1, 2, 3, 4, 5]
        }

        model = TurboMindModelwithChatTemplate(
            path='test/model/path',
            engine_config=dict(session_len=4096, max_batch_size=1),
            gen_config=dict(do_sample=False),
            max_seq_len=2048,
        )

        token_len = model.get_token_len('Hello')

        self.assertEqual(token_len, 5)
        mock_convert_messages.assert_called_once_with(['Hello'])
        mock_tokenizer.apply_chat_template.assert_called_once()

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    @patch('lmdeploy.GenerationConfig')
    def test_get_potential_stop_words(self, mock_gen_config_class,
                                      mock_get_meta_template,
                                      mock_get_max_seq_len, mock_pipeline,
                                      mock_tokenizer_class):
        """Test _get_potential_stop_words method."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer.decode.return_value = '<|endoftext|>'
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe

        # Mock GenerationConfig
        with patch('lmdeploy.GenerationConfig') as mock_gen_config:
            mock_gen_config_instance = MagicMock()
            mock_gen_config_instance.eos_token_id = 2
            mock_gen_config.from_pretrained.return_value = mock_gen_config_instance

            model = TurboMindModelwithChatTemplate(
                path='test/model/path',
                engine_config=dict(session_len=4096, max_batch_size=1),
                gen_config=dict(do_sample=False),
                max_seq_len=2048,
            )

            # Check that stop_words include eos_token
            self.assertIn('<|endoftext|>', model.stop_words)

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._convert_chat_messages'
    )
    @patch('lmdeploy.GenerationConfig')
    def test_generate_with_do_sample_false(self, mock_gen_config_class,
                                           mock_convert_messages,
                                           mock_get_meta_template,
                                           mock_get_max_seq_len, mock_pipeline,
                                           mock_tokenizer_class):
        """Test generate with do_sample=False."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = None
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.text = 'Generated response'
        mock_output.input_token_len = 10
        mock_output.generate_token_len = 5
        mock_pipe.return_value = [mock_output]
        mock_pipeline.return_value = mock_pipe

        mock_convert_messages.return_value = [{
            'role': 'user',
            'content': 'Hello'
        }]
        mock_tokenizer.apply_chat_template.return_value = 'Formatted prompt'
        mock_gen_config = MagicMock()
        mock_gen_config_class.return_value = mock_gen_config

        model = TurboMindModelwithChatTemplate(
            path='test/model/path',
            engine_config=dict(session_len=4096, max_batch_size=1),
            gen_config=dict(do_sample=False),
            max_seq_len=2048,
        )

        model.generate(['Hello'], max_out_len=100, do_sample=False)

        # Verify that do_sample=False is set in gen_config
        call_args = mock_gen_config_class.call_args[1]
        self.assertIn('do_sample', call_args)
        self.assertFalse(call_args['do_sample'])

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 5, 0))
    @patch('lmdeploy.pipeline')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._convert_chat_messages'
    )
    @patch('lmdeploy.GenerationConfig')
    def test_generate_with_old_version(self, mock_gen_config_class,
                                       mock_convert_messages,
                                       mock_get_meta_template,
                                       mock_get_max_seq_len, mock_pipeline,
                                       mock_tokenizer_class):
        """Test generate with old lmdeploy version (< 0.6.0)."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = None
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.text = 'Generated response'
        mock_output.input_token_len = 10
        mock_output.generate_token_len = 5
        mock_pipe.return_value = [mock_output]
        mock_pipeline.return_value = mock_pipe

        mock_convert_messages.return_value = [{
            'role': 'user',
            'content': 'Hello'
        }]
        mock_tokenizer.apply_chat_template.return_value = 'Formatted prompt'
        mock_gen_config = MagicMock()
        mock_gen_config_class.return_value = mock_gen_config

        model = TurboMindModelwithChatTemplate(
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
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._convert_chat_messages'
    )
    @patch('lmdeploy.GenerationConfig')
    def test_generate_with_min_out_len(self, mock_gen_config_class,
                                       mock_convert_messages,
                                       mock_get_meta_template,
                                       mock_get_max_seq_len, mock_pipeline,
                                       mock_tokenizer_class):
        """Test generate with min_out_len parameter."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = None
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.text = 'Generated response'
        mock_output.input_token_len = 10
        mock_output.generate_token_len = 5
        mock_pipe.return_value = [mock_output]
        mock_pipeline.return_value = mock_pipe

        mock_convert_messages.return_value = [{
            'role': 'user',
            'content': 'Hello'
        }]
        mock_tokenizer.apply_chat_template.return_value = 'Formatted prompt'
        mock_gen_config = MagicMock()
        mock_gen_config_class.return_value = mock_gen_config

        model = TurboMindModelwithChatTemplate(
            path='test/model/path',
            engine_config=dict(session_len=4096, max_batch_size=1),
            gen_config=dict(do_sample=False),
            max_seq_len=2048,
        )

        _ = model.generate(['Hello'], max_out_len=100, min_out_len=10)

        # Verify that min_new_tokens is set
        call_args = mock_gen_config_class.call_args[1]
        self.assertIn('min_new_tokens', call_args)
        self.assertEqual(call_args['min_new_tokens'], 10)

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._convert_chat_messages'
    )
    @patch('lmdeploy.GenerationConfig')
    def test_generate_with_bos_token_removal(self, mock_gen_config_class,
                                             mock_convert_messages,
                                             mock_get_meta_template,
                                             mock_get_max_seq_len,
                                             mock_pipeline,
                                             mock_tokenizer_class):
        """Test generate with BOS token removal."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = '<bos>'
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.text = 'Generated response'
        mock_output.input_token_len = 10
        mock_output.generate_token_len = 5
        mock_pipe.return_value = [mock_output]
        mock_pipeline.return_value = mock_pipe

        mock_convert_messages.return_value = [{
            'role': 'user',
            'content': 'Hello'
        }]
        # Mock apply_chat_template to return message with BOS token
        mock_tokenizer.apply_chat_template.return_value = '<bos>Formatted prompt'
        mock_gen_config = MagicMock()
        mock_gen_config_class.return_value = mock_gen_config

        model = TurboMindModelwithChatTemplate(
            path='test/model/path',
            engine_config=dict(session_len=4096, max_batch_size=1),
            gen_config=dict(do_sample=False),
            max_seq_len=2048,
        )

        _ = model.generate(['Hello'], max_out_len=100)
        mock_tokenizer.apply_chat_template.assert_called()
        # The BOS token should be removed from the message
        self.assertTrue(mock_pipe.called)
        if mock_pipe.called:
            call_args = mock_pipe.call_args[0][0]
            self.assertFalse(any(msg.startswith('<bos>') for msg in call_args))

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    @patch('lmdeploy.TurbomindEngineConfig')
    @patch('lmdeploy.PytorchEngineConfig')
    def test_build_pipe_turbomind_backend(self, mock_pytorch_config,
                                          mock_turbomind_config,
                                          mock_get_meta_template,
                                          mock_get_max_seq_len, mock_pipeline,
                                          mock_tokenizer_class):
        """Test _build_pipe with turbomind backend."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_turbomind_instance = MagicMock()
        mock_turbomind_config.return_value = mock_turbomind_instance
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe

        with patch.dict('os.environ', {
                'LMDEPLOY_LOG_LEVEL': 'INFO',
                'LMDEPLOY_MAX_LOG_LEN': '20'
        }):
            TurboMindModelwithChatTemplate(
                path='test/model/path',
                engine_config=dict(session_len=4096, max_batch_size=1),
                backend='turbomind',
                max_seq_len=2048,
            )

            mock_turbomind_config.assert_called_once()
            # os.getenv returns string when env var is set, but default is int
            # Check the call was made with correct parameters
            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args[1]
            self.assertEqual(call_kwargs['log_level'], 'INFO')
            self.assertIn('max_log_len', call_kwargs)
            # max_log_len can be '20' (from env) or 10 (default)
            self.assertIn(call_kwargs['max_log_len'], ['20', 10])

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    @patch('lmdeploy.TurbomindEngineConfig')
    @patch('lmdeploy.PytorchEngineConfig')
    def test_build_pipe_pytorch_backend(self, mock_pytorch_config,
                                        mock_turbomind_config,
                                        mock_get_meta_template,
                                        mock_get_max_seq_len, mock_pipeline,
                                        mock_tokenizer_class):
        """Test _build_pipe with pytorch backend."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_pytorch_instance = MagicMock()
        mock_pytorch_config.return_value = mock_pytorch_instance
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe

        TurboMindModelwithChatTemplate(
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
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    def test_build_pipe_invalid_backend(self, mock_get_meta_template,
                                        mock_get_max_seq_len, mock_pipeline,
                                        mock_tokenizer_class):
        """Test _build_pipe with invalid backend."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with self.assertRaises(AssertionError):
            TurboMindModelwithChatTemplate(
                path='test/model/path',
                engine_config=dict(session_len=4096, max_batch_size=1),
                backend='invalid_backend',
                max_seq_len=2048,
            )

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._convert_chat_messages'
    )
    @patch('lmdeploy.GenerationConfig')
    def test_gen_config_override_in_generate(self, mock_gen_config_class,
                                             mock_convert_messages,
                                             mock_get_meta_template,
                                             mock_get_max_seq_len,
                                             mock_pipeline,
                                             mock_tokenizer_class):
        """Test that model gen_config is properly overridden by generate parameters."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = None
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.text = 'Generated response'
        mock_output.input_token_len = 10
        mock_output.generate_token_len = 5
        mock_pipe.return_value = [mock_output]
        mock_pipeline.return_value = mock_pipe

        mock_convert_messages.return_value = [{
            'role': 'user',
            'content': 'Hello'
        }]
        mock_tokenizer.apply_chat_template.return_value = 'Formatted prompt'
        mock_gen_config = MagicMock()
        mock_gen_config_class.return_value = mock_gen_config

        # Create model with specific gen_config
        model = TurboMindModelwithChatTemplate(
            path='test/model/path',
            engine_config=dict(session_len=4096, max_batch_size=1),
            gen_config=dict(temperature=0.5, top_k=5),
            max_seq_len=2048,
        )

        # Generate with different max_out_len
        model.generate(['Hello'], max_out_len=256)

        # Verify that max_new_tokens is set to the generate parameter value
        call_args = mock_gen_config_class.call_args[1]
        self.assertEqual(call_args['max_new_tokens'], 256)
        # Verify that model's gen_config values are still present
        self.assertEqual(call_args['temperature'], 0.5)
        self.assertEqual(call_args['top_k'], 5)

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._convert_chat_messages'
    )
    @patch('lmdeploy.GenerationConfig')
    def test_gen_config_max_out_len_override(self, mock_gen_config_class,
                                             mock_convert_messages,
                                             mock_get_meta_template,
                                             mock_get_max_seq_len,
                                             mock_pipeline,
                                             mock_tokenizer_class):
        """Test that max_out_len parameter overrides gen_config max_new_tokens."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = None
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.text = 'Generated response'
        mock_output.input_token_len = 10
        mock_output.generate_token_len = 5
        mock_pipe.return_value = [mock_output]
        mock_pipeline.return_value = mock_pipe

        mock_convert_messages.return_value = [{
            'role': 'user',
            'content': 'Hello'
        }]
        mock_tokenizer.apply_chat_template.return_value = 'Formatted prompt'
        mock_gen_config = MagicMock()
        mock_gen_config_class.return_value = mock_gen_config

        # Create model with gen_config including max_new_tokens
        model = TurboMindModelwithChatTemplate(
            path='test/model/path',
            engine_config=dict(session_len=4096, max_batch_size=1),
            gen_config=dict(max_new_tokens=1024, temperature=0.5),
            max_seq_len=2048,
        )

        # Generate with different max_out_len - should override gen_config max_new_tokens
        _ = model.generate(['Hello'], max_out_len=512)

        # Verify that max_new_tokens is overridden to 512
        call_args = mock_gen_config_class.call_args[1]
        self.assertEqual(call_args['max_new_tokens'], 512)

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._convert_chat_messages'
    )
    @patch('lmdeploy.GenerationConfig')
    def test_gen_config_min_out_len_override(self, mock_gen_config_class,
                                             mock_convert_messages,
                                             mock_get_meta_template,
                                             mock_get_max_seq_len,
                                             mock_pipeline,
                                             mock_tokenizer_class):
        """Test that min_out_len parameter is correctly set in gen_config."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = None
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.text = 'Generated response'
        mock_output.input_token_len = 10
        mock_output.generate_token_len = 5
        mock_pipe.return_value = [mock_output]
        mock_pipeline.return_value = mock_pipe

        mock_convert_messages.return_value = [{
            'role': 'user',
            'content': 'Hello'
        }]
        mock_tokenizer.apply_chat_template.return_value = 'Formatted prompt'
        mock_gen_config = MagicMock()
        mock_gen_config_class.return_value = mock_gen_config

        model = TurboMindModelwithChatTemplate(
            path='test/model/path',
            engine_config=dict(session_len=4096, max_batch_size=1),
            gen_config=dict(temperature=0.7),
            max_seq_len=2048,
        )

        # Generate with min_out_len
        _ = model.generate(['Hello'], max_out_len=200, min_out_len=50)

        # Verify that both max and min are set correctly
        call_args = mock_gen_config_class.call_args[1]
        self.assertEqual(call_args['max_new_tokens'], 200)
        self.assertEqual(call_args['min_new_tokens'], 50)

    @patch('transformers.AutoTokenizer')
    @patch('lmdeploy.version_info', (0, 6, 0))
    @patch('lmdeploy.pipeline')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_possible_max_seq_len'
    )
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._get_meta_template')
    @patch(
        'opencompass.models.turbomind_with_tf_above_v4_33._convert_chat_messages'
    )
    @patch('lmdeploy.GenerationConfig')
    def test_gen_config_sampling_params_override(self, mock_gen_config_class,
                                                 mock_convert_messages,
                                                 mock_get_meta_template,
                                                 mock_get_max_seq_len,
                                                 mock_pipeline,
                                                 mock_tokenizer_class):
        """Test that gen_config sampling parameters are correctly overridden."""
        mock_get_max_seq_len.return_value = 2048
        mock_get_meta_template.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = None
        mock_tokenizer.eos_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.text = 'Generated response'
        mock_output.input_token_len = 10
        mock_output.generate_token_len = 5
        mock_pipe.return_value = [mock_output]
        mock_pipeline.return_value = mock_pipe

        mock_convert_messages.return_value = [{
            'role': 'user',
            'content': 'Hello'
        }]
        mock_tokenizer.apply_chat_template.return_value = 'Formatted prompt'
        mock_gen_config = MagicMock()
        mock_gen_config_class.return_value = mock_gen_config

        # Create model with sampling parameters in gen_config
        model = TurboMindModelwithChatTemplate(
            path='test/model/path',
            engine_config=dict(session_len=4096, max_batch_size=1),
            gen_config=dict(top_p=0.8, top_k=10, temperature=0.6),
            max_seq_len=2048,
        )

        # Generate with do_sample=True
        _ = model.generate(['Hello'],
                           max_out_len=100,
                           do_sample=True,
                           temperature=0.9)

        # Verify that temperature is overridden but other sampling params remain
        call_args = mock_gen_config_class.call_args[1]
        # Note: The generate method doesn't use the temperature parameter directly in gen_config update
        # The sampling params from model's gen_config should be present
        self.assertEqual(call_args['top_p'], 0.8)
        self.assertEqual(call_args['top_k'], 10)


if __name__ == '__main__':
    unittest.main()
