"""Unit tests for InferStatusManager."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mmengine.config import ConfigDict

from opencompass.utils.infer_status import (InferStatusManager, safe_read,
                                            safe_write)


class TestSafeReadWrite(unittest.TestCase):
    """Test cases for safe_read and safe_write functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_file = self.temp_dir / 'test.json'
        self.test_file.write_text('{"test": "data"}')

    def test_safe_read(self):
        """Test safe_read function."""
        result = safe_read(self.test_file, self.temp_dir)
        self.assertEqual(result, '{"test": "data"}')

    def test_safe_write(self):
        """Test safe_write function."""
        new_content = '{"new": "content"}'
        safe_write(self.test_file, new_content, self.temp_dir)
        self.assertEqual(self.test_file.read_text(), new_content)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


class TestInferStatusManager(unittest.TestCase):
    """Test cases for InferStatusManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.model_cfg = ConfigDict({'abbr': 'test_model'})
        self.dataset_cfg = ConfigDict({'abbr': 'test_dataset'})

    def test_initialization(self):
        """Test InferStatusManager initialization."""
        manager = InferStatusManager(self.temp_dir, self.model_cfg,
                                     self.dataset_cfg)
        self.assertEqual(manager.model_cfg, self.model_cfg)
        self.assertEqual(manager.dataset_cfg, self.dataset_cfg)
        self.assertEqual(manager.work_dir, self.temp_dir)
        self.assertEqual(manager.infer_status['status'], 'pending')
        self.assertEqual(manager.infer_status['completed'], 0)
        self.assertIsNone(manager.infer_status['total'])
        self.assertIsNotNone(manager.status_path)

    def test_update_status(self):
        """Test update method with status."""
        manager = InferStatusManager(self.temp_dir, self.model_cfg,
                                     self.dataset_cfg)
        manager.update(status='running')
        self.assertEqual(manager.infer_status['status'], 'running')

    def test_update_total(self):
        """Test update method with total."""
        manager = InferStatusManager(self.temp_dir, self.model_cfg,
                                     self.dataset_cfg)
        manager.update(total=100)
        self.assertEqual(manager.infer_status['total'], 100)

    def test_update_completed(self):
        """Test update method with completed."""
        manager = InferStatusManager(self.temp_dir, self.model_cfg,
                                     self.dataset_cfg)
        manager.update(completed=50)
        self.assertEqual(manager.infer_status['completed'], 50)

    def test_update_multiple(self):
        """Test update method with multiple parameters."""
        manager = InferStatusManager(self.temp_dir, self.model_cfg,
                                     self.dataset_cfg)
        manager.update(status='done', total=100, completed=100)
        self.assertEqual(manager.infer_status['status'], 'done')
        self.assertEqual(manager.infer_status['total'], 100)
        self.assertEqual(manager.infer_status['completed'], 100)

    def test_update_partial(self):
        """Test update method with partial parameters."""
        manager = InferStatusManager(self.temp_dir, self.model_cfg,
                                     self.dataset_cfg)
        manager.update(status='running', total=100)
        manager.update(completed=50)
        self.assertEqual(manager.infer_status['status'], 'running')
        self.assertEqual(manager.infer_status['total'], 100)
        self.assertEqual(manager.infer_status['completed'], 50)

    def test_write_task_status(self):
        """Test write_task_status method."""
        manager = InferStatusManager(self.temp_dir, self.model_cfg,
                                     self.dataset_cfg)
        manager.update(status='done', total=100, completed=100)
        manager.write_task_status()

        # Verify file was created
        self.assertTrue(manager.status_path.exists())
        content = json.loads(manager.status_path.read_text())
        self.assertEqual(content['status'], 'done')
        self.assertEqual(content['total'], 100)
        self.assertEqual(content['completed'], 100)
        self.assertIn('updated_at', content)

    def test_get_task_status_existing(self):
        """Test get_task_status with existing file."""
        manager = InferStatusManager(self.temp_dir, self.model_cfg,
                                     self.dataset_cfg)
        manager.update(status='done', total=100, completed=100)
        manager.write_task_status()

        status = manager.get_task_status()
        self.assertIsInstance(status, dict)
        self.assertIn(manager.status_path.stem, status)
        self.assertEqual(status[manager.status_path.stem]['status'], 'done')

    def test_get_task_status_nonexistent(self):
        """Test get_task_status with non-existent file."""
        manager = InferStatusManager(self.temp_dir, self.model_cfg,
                                     self.dataset_cfg)
        status = manager.get_task_status()
        self.assertEqual(status, {})

    def test_get_task_status_with_children(self):
        """Test get_task_status with child files."""
        manager = InferStatusManager(self.temp_dir, self.model_cfg,
                                     self.dataset_cfg)
        # Create child files
        child1 = manager.status_path.parent / f'{manager.status_path.stem}_0.json'  # noqa
        child2 = manager.status_path.parent / f'{manager.status_path.stem}_1.json'  # noqa
        child1.write_text(
            json.dumps({
                'status': 'done',
                'total': 50,
                'completed': 50
            }))
        child2.write_text(
            json.dumps({
                'status': 'done',
                'total': 50,
                'completed': 50
            }))

        status = manager.get_task_status()
        self.assertIsInstance(status, dict)
        self.assertIn(f'{manager.status_path.stem}_0', status)
        self.assertIn(f'{manager.status_path.stem}_1', status)

    def test_maybe_write_only_on_change(self):
        """Test _maybe_write only writes when status changes."""
        manager = InferStatusManager(self.temp_dir, self.model_cfg,
                                     self.dataset_cfg)

        # First update should write
        with patch.object(manager, 'write_task_status') as mock_write:
            manager.update(status='running')
            mock_write.assert_called_once()

        # Same update should not write again
        with patch.object(manager, 'write_task_status') as mock_write:
            manager.update(status='running')
            mock_write.assert_not_called()

        # Different update should write
        with patch.object(manager, 'write_task_status') as mock_write:
            manager.update(status='done')
            mock_write.assert_called_once()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()
