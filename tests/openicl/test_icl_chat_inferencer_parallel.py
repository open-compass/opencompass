"""Unit tests for ParallelChatInferencer."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from opencompass.openicl.icl_inferencer.icl_chat_inferencer_parallel import \
    ParallelChatInferencer


class TestParallelChatInferencer(unittest.TestCase):
    """Test cases for ParallelChatInferencer."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_model = MagicMock()
        self.mock_model.is_api = True
        # Add template_parser to avoid AttributeError
        self.mock_model.template_parser = MagicMock()

    def test_initialization(self):
        """Test ParallelChatInferencer initialization."""
        inferencer = ParallelChatInferencer(model=self.mock_model,
                                            max_out_len=512,
                                            max_infer_workers=4)
        self.assertEqual(inferencer.max_infer_workers, 4)
        self.assertIsNone(inferencer.progress_tracker)
        self.assertEqual(inferencer.max_out_len, 512)

    def test_initialization_defaults(self):
        """Test initialization with default values."""
        inferencer = ParallelChatInferencer(model=self.mock_model,
                                            max_out_len=512)
        self.assertIsNone(inferencer.max_infer_workers)
        self.assertIsNone(inferencer.progress_tracker)

    def test_resolve_max_workers_from_config(self):
        """Test _resolve_max_workers uses max_infer_workers."""
        inferencer = ParallelChatInferencer(model=self.mock_model,
                                            max_out_len=512,
                                            max_infer_workers=8)
        result = inferencer._resolve_max_workers()
        self.assertEqual(result, 8)

    def test_resolve_max_workers_from_model(self):
        """Test _resolve_max_workers uses model.max_workers."""
        self.mock_model.max_workers = 6
        inferencer = ParallelChatInferencer(model=self.mock_model,
                                            max_out_len=512)
        result = inferencer._resolve_max_workers()
        self.assertEqual(result, 6)

    @patch(
        'opencompass.openicl.icl_inferencer.icl_chat_inferencer_parallel.os.cpu_count'  # noqa
    )
    @patch(
        'opencompass.openicl.icl_inferencer.icl_chat_inferencer_parallel.getattr'  # noqa
    )
    def test_resolve_max_workers_default(self, mock_getattr, mock_cpu_count):
        """Test _resolve_max_workers uses default calculation."""
        mock_cpu_count.return_value = 8

        # Make getattr return None for max_workers
        def getattr_side_effect(obj, attr, default=None):
            if attr == 'max_workers':
                return None
            return getattr(obj, attr, default)

        mock_getattr.side_effect = getattr_side_effect

        inferencer = ParallelChatInferencer(model=self.mock_model,
                                            max_out_len=512)
        inferencer.max_infer_workers = None
        result = inferencer._resolve_max_workers()
        # Should be min(32, cpu_count + 4) = min(32, 12) = 12
        self.assertEqual(result, 12)

    @patch(
        'opencompass.openicl.icl_inferencer.icl_chat_inferencer_parallel.os.cpu_count'  # noqa
    )
    @patch(
        'opencompass.openicl.icl_inferencer.icl_chat_inferencer_parallel.getattr'  # noqa
    )
    def test_resolve_max_workers_max_limit(self, mock_getattr, mock_cpu_count):
        """Test _resolve_max_workers respects max limit."""
        mock_cpu_count.return_value = 100

        # Make getattr return None for max_workers
        def getattr_side_effect(obj, attr, default=None):
            if attr == 'max_workers':
                return None
            return getattr(obj, attr, default)

        mock_getattr.side_effect = getattr_side_effect

        inferencer = ParallelChatInferencer(model=self.mock_model,
                                            max_out_len=512)
        inferencer.max_infer_workers = None
        result = inferencer._resolve_max_workers()
        # Should be min(32, 100 + 4) = 32
        self.assertEqual(result, 32)

    def test_progress_update(self):
        """Test _progress_update method."""
        inferencer = ParallelChatInferencer(model=self.mock_model,
                                            max_out_len=512)
        mock_tracker = MagicMock()
        inferencer.progress_tracker = mock_tracker

        inferencer._progress_update(5)
        mock_tracker.incr.assert_called_once_with(5)

    def test_progress_update_no_tracker(self):
        """Test _progress_update when tracker is None."""
        inferencer = ParallelChatInferencer(model=self.mock_model,
                                            max_out_len=512)
        # Should not raise error
        inferencer._progress_update(5)

    @patch(
        'opencompass.openicl.icl_inferencer.icl_chat_inferencer_parallel.os.path.exists'  # noqa
    )
    @patch(
        'opencompass.openicl.icl_inferencer.icl_chat_inferencer_parallel.os.makedirs'  # noqa
    )
    @patch.object(ParallelChatInferencer,
                  '_resolve_max_workers',
                  return_value=2)
    def test_inference_basic(self, mock_resolve, mock_makedirs, mock_exists):
        """Test inference method basic functionality."""
        mock_exists.return_value = False

        inferencer = ParallelChatInferencer(model=self.mock_model,
                                            max_out_len=512,
                                            output_json_filepath=self.temp_dir,
                                            output_json_filename='test.json')

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [[0], [1]]

        # Mock the get_chat_list method
        inferencer.get_chat_list = MagicMock(
            return_value=[['chat1'], ['chat2']])

        # Mock HandlerType and infer methods
        mock_handler = MagicMock()
        mock_handler.results_dict = {}
        inferencer.HandlerType = MagicMock(return_value=mock_handler)
        inferencer.infer_last = MagicMock()

        result = inferencer.inference(mock_retriever)

        self.assertIsInstance(result, dict)
        mock_retriever.retrieve.assert_called_once()

    @patch(
        'opencompass.openicl.icl_inferencer.icl_chat_inferencer_parallel.os.path.exists'  # noqa
    )
    @patch(
        'opencompass.openicl.icl_inferencer.icl_chat_inferencer_parallel.os.makedirs'  # noqa
    )
    @patch.object(ParallelChatInferencer,
                  '_resolve_max_workers',
                  return_value=2)
    def test_inference_with_progress_tracker(self, mock_resolve, mock_makedirs,
                                             mock_exists):
        """Test inference with progress tracker."""
        mock_exists.return_value = False

        inferencer = ParallelChatInferencer(model=self.mock_model,
                                            max_out_len=512,
                                            output_json_filepath=self.temp_dir,
                                            output_json_filename='test.json')

        mock_tracker = MagicMock()
        inferencer.progress_tracker = mock_tracker

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [[0], [1]]

        inferencer.get_chat_list = MagicMock(
            return_value=[['chat1'], ['chat2']])

        mock_handler = MagicMock()
        mock_handler.results_dict = {}
        inferencer.HandlerType = MagicMock(return_value=mock_handler)
        inferencer.infer_last = MagicMock()

        inferencer.inference(mock_retriever)

        # Verify tracker was used
        mock_tracker.set_total.assert_called_once_with(2)

    @patch(
        'opencompass.openicl.icl_inferencer.icl_chat_inferencer_parallel.os.path.exists'  # noqa
    )
    @patch(
        'opencompass.openicl.icl_inferencer.icl_chat_inferencer_parallel.os.makedirs'  # noqa
    )
    @patch(
        'opencompass.openicl.icl_inferencer.icl_chat_inferencer_parallel.os.remove'  # noqa 
    )
    @patch.object(ParallelChatInferencer,
                  '_resolve_max_workers',
                  return_value=2)
    def test_inference_with_resume(self, mock_resolve, mock_remove,
                                   mock_makedirs, mock_exists):
        """Test inference with resume from tmp file."""

        # Make exists return True for tmp file check, False for removal check
        def exists_side_effect(path):
            if 'tmp_' in str(path):
                return True
            return False

        mock_exists.side_effect = exists_side_effect

        inferencer = ParallelChatInferencer(model=self.mock_model,
                                            max_out_len=512,
                                            output_json_filepath=self.temp_dir,
                                            output_json_filename='test.json')

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [[0], [1]]

        inferencer.get_chat_list = MagicMock(
            return_value=[['chat1'], ['chat2']])

        mock_handler = MagicMock()
        mock_handler.results_dict = {'0': {'prediction': 'existing'}}
        mock_handler.restore_from_jsonl = MagicMock(
            return_value={'0': {
                'prediction': 'existing'
            }})
        inferencer.HandlerType = MagicMock(return_value=mock_handler)
        inferencer.infer_last = MagicMock()

        inferencer.inference(mock_retriever)

        mock_retriever.retrieve.assert_called_once()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()
