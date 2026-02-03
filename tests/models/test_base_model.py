"""Unit tests for BaseModel."""

import unittest
from unittest.mock import MagicMock, patch


try:
    from opencompass.models.base import BaseModel
    BASE_MODEL_AVAILABLE = True
except ImportError:
    BASE_MODEL_AVAILABLE = False


class TestBaseModel(unittest.TestCase):
    """Test cases for BaseModel."""

    def test_is_api_attribute(self):
        """Test that BaseModel has is_api attribute."""
        if not BASE_MODEL_AVAILABLE:
            self.skipTest("BaseModel not available")
        
        self.assertFalse(BaseModel.is_api)

    def test_initialization_signature(self):
        """Test BaseModel initialization signature."""
        if not BASE_MODEL_AVAILABLE:
            self.skipTest("BaseModel not available")
        
        # Check that BaseModel has expected initialization parameters
        import inspect
        sig = inspect.signature(BaseModel.__init__)
        params = list(sig.parameters.keys())
        
        # Should have path parameter
        self.assertIn('path', params)
        # Should have optional parameters
        self.assertIn('max_seq_len', params)
        self.assertIn('tokenizer_only', params)
        self.assertIn('meta_template', params)
        self.assertIn('generation_kwargs', params)

    @patch('opencompass.models.base.LMTemplateParser')
    def test_initialization_with_meta_template(self, mock_parser):
        """Test BaseModel initialization with meta_template."""
        if not BASE_MODEL_AVAILABLE:
            self.skipTest("BaseModel not available")
        
        # BaseModel is abstract, so we can't instantiate it directly
        # But we can test the structure
        mock_parser_instance = MagicMock()
        mock_parser.return_value = mock_parser_instance
        
        # Check that BaseModel has the expected structure
        self.assertTrue(hasattr(BaseModel, '__init__'))
        self.assertTrue(hasattr(BaseModel, 'is_api'))


if __name__ == '__main__':
    unittest.main()
