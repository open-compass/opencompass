"""Unit tests for BaseTask."""

import tempfile
import unittest

from mmengine.config import ConfigDict

from opencompass.tasks.base import BaseTask, extract_role_pred


class TestExtractRolePred(unittest.TestCase):
    """Test cases for extract_role_pred function."""

    def test_extract_role_pred_function_source(self):
        """Test that extract_role_pred uses the fixed implementation."""

        import inspect

        import opencompass.tasks.base as base_module

        # Check which module is being used
        module_path = base_module.__file__
        source = inspect.getsource(extract_role_pred)

        # Verify the fix is present (using .strip() instead of re.match)
        if 'begin_str.strip()' not in source:
            self.fail(f'Function should use begin_str.strip(). '
                      f'Module loaded from: {module_path}. '
                      f'Source: {source[:200]}')
        if 'end_str.strip()' not in source:
            self.fail(f'Function should use end_str.strip(). '
                      f'Module loaded from: {module_path}')

        # Verify old buggy code is not present
        if "re.match(r'\\s*', begin_str)" in source:  # noqa
            self.fail(f'Function should not use the buggy re.match check. '
                      f'Module loaded from: {module_path}')

    def deperacated_test_extract_role_pred_with_begin_and_end(self):
        """Test extract_role_pred with begin and end strings."""

        text = 'prefix ANSWER: test answer SUFFIX'
        result = extract_role_pred(text,
                                   begin_str='ANSWER: ',
                                   end_str=' SUFFIX')
        # Verify that extraction works correctly
        self.assertEqual(result, 'test answer',
                         f'Expected \'test answer\', got {repr(result)}')

    def deperacated_test_extract_role_pred_with_begin_only(self):
        """Test extract_role_pred with begin string only."""

        text = 'prefix ANSWER: test answer'
        result = extract_role_pred(text, begin_str='ANSWER: ', end_str=None)
        # Verify that extraction works correctly
        self.assertEqual(result, 'test answer',
                         f'Expected \'test answer\', got {repr(result)}')

    def deperacated_test_extract_role_pred_with_end_only(self):
        """Test extract_role_pred with end string only."""

        text = 'test answer SUFFIX'
        result = extract_role_pred(text, begin_str=None, end_str=' SUFFIX')
        # Verify that extraction works correctly
        self.assertEqual(result, 'test answer',
                         f'Expected \'test answer\', got {repr(result)}')

    def test_extract_role_pred_without_markers(self):
        """Test extract_role_pred without begin/end markers."""

        text = 'test answer'
        result = extract_role_pred(text, begin_str=None, end_str=None)
        self.assertEqual(result, 'test answer')

    def test_extract_role_pred_with_whitespace_begin(self):
        """Test extract_role_pred with whitespace-only begin string."""

        text = 'test answer'
        result = extract_role_pred(text, begin_str='   ', end_str=None)
        # Whitespace-only begin_str should be ignored
        self.assertEqual(result, 'test answer')


class TestBaseTask(unittest.TestCase):
    """Test cases for BaseTask."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = ConfigDict({
            'work_dir':
            tempfile.mkdtemp(),
            'models': [ConfigDict({'abbr': 'test_model'})],
            'datasets': [ConfigDict({'abbr': 'test_dataset'})]
        })

    def test_initialization(self):
        """Test BaseTask initialization."""

        # BaseTask is abstract, so we can't instantiate it directly
        # We'll test that it can be imported and has the expected structure
        self.assertTrue(hasattr(BaseTask, '__init__'))
        self.assertTrue(hasattr(BaseTask, 'run'))
        self.assertTrue(hasattr(BaseTask, 'get_command'))


if __name__ == '__main__':
    unittest.main()
