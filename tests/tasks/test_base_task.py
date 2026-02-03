"""Unit tests for BaseTask."""

import tempfile
import unittest
from unittest.mock import MagicMock, patch

from mmengine.config import ConfigDict


try:
    from opencompass.tasks.base import BaseTask, extract_role_pred
    BASE_TASK_AVAILABLE = True
except ImportError:
    BASE_TASK_AVAILABLE = False


class TestExtractRolePred(unittest.TestCase):
    """Test cases for extract_role_pred function."""

    def test_extract_role_pred_with_begin_and_end(self):
        """Test extract_role_pred with begin and end strings."""
        if not BASE_TASK_AVAILABLE:
            self.skipTest("extract_role_pred not available")
        
        text = "prefix ANSWER: test answer SUFFIX"
        result = extract_role_pred(text, begin_str="ANSWER: ", end_str=" SUFFIX")
        self.assertEqual(result, "test answer")

    def test_extract_role_pred_with_begin_only(self):
        """Test extract_role_pred with begin string only."""
        if not BASE_TASK_AVAILABLE:
            self.skipTest("extract_role_pred not available")
        
        text = "prefix ANSWER: test answer"
        result = extract_role_pred(text, begin_str="ANSWER: ", end_str=None)
        self.assertEqual(result, "test answer")

    def test_extract_role_pred_with_end_only(self):
        """Test extract_role_pred with end string only."""
        if not BASE_TASK_AVAILABLE:
            self.skipTest("extract_role_pred not available")
        
        text = "test answer SUFFIX"
        result = extract_role_pred(text, begin_str=None, end_str=" SUFFIX")
        self.assertEqual(result, "test answer")

    def test_extract_role_pred_without_markers(self):
        """Test extract_role_pred without begin/end markers."""
        if not BASE_TASK_AVAILABLE:
            self.skipTest("extract_role_pred not available")
        
        text = "test answer"
        result = extract_role_pred(text, begin_str=None, end_str=None)
        self.assertEqual(result, "test answer")

    def test_extract_role_pred_with_whitespace_begin(self):
        """Test extract_role_pred with whitespace-only begin string."""
        if not BASE_TASK_AVAILABLE:
            self.skipTest("extract_role_pred not available")
        
        text = "test answer"
        result = extract_role_pred(text, begin_str="   ", end_str=None)
        # Whitespace-only begin_str should be ignored
        self.assertEqual(result, "test answer")


class TestBaseTask(unittest.TestCase):
    """Test cases for BaseTask."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = ConfigDict({
            'work_dir': tempfile.mkdtemp(),
            'models': [ConfigDict({'abbr': 'test_model'})],
            'datasets': [ConfigDict({'abbr': 'test_dataset'})]
        })

    def test_initialization(self):
        """Test BaseTask initialization."""
        if not BASE_TASK_AVAILABLE:
            self.skipTest("BaseTask not available")
        
        # BaseTask is abstract, so we can't instantiate it directly
        # We'll test that it can be imported and has the expected structure
        self.assertTrue(hasattr(BaseTask, '__init__'))
        self.assertTrue(hasattr(BaseTask, 'run'))
        self.assertTrue(hasattr(BaseTask, 'get_command'))


if __name__ == '__main__':
    unittest.main()
