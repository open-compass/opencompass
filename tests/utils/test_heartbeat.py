"""Unit tests for HeartBeatManager."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from opencompass.utils.heartbeat import HeartBeatManager


class TestHeartBeatManager(unittest.TestCase):
    """Test cases for HeartBeatManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def test_initialization(self):
        """Test HeartBeatManager initialization."""
        manager = HeartBeatManager(self.temp_dir)
        self.assertEqual(manager.hb_file, self.temp_dir / 'infer_heartbeat')
        self.assertIsNotNone(manager._lock)

    def test_initialization_with_custom_fname(self):
        """Test HeartBeatManager initialization with custom filename."""
        manager = HeartBeatManager(self.temp_dir, fname='custom_heartbeat')
        self.assertEqual(manager.hb_file, self.temp_dir / 'custom_heartbeat')

    def test_last_heartbeat_nonexistent(self):
        """Test last_heartbeat when file doesn't exist."""
        manager = HeartBeatManager(self.temp_dir)
        result = manager.last_heartbeat()
        self.assertEqual(result, float('inf'))

    def test_last_heartbeat_existing(self):
        """Test last_heartbeat when file exists."""
        manager = HeartBeatManager(self.temp_dir)
        # Create heartbeat file
        manager.hb_file.parent.mkdir(parents=True, exist_ok=True)
        manager.hb_file.write_text('2024-01-01T00:00:00')

        # Mock datetime.now to return a time 10 seconds later
        with patch('opencompass.utils.heartbeat.datetime') as mock_datetime:
            from datetime import datetime
            mock_datetime.now.return_value = datetime.fromisoformat(
                '2024-01-01T00:00:10')
            mock_datetime.fromisoformat.return_value = datetime.fromisoformat(
                '2024-01-01T00:00:00')

            result = manager.last_heartbeat()
            self.assertAlmostEqual(result, 10.0, places=1)

    def test_start_heartbeat(self):
        """Test start_heartbeat method."""
        manager = HeartBeatManager(self.temp_dir)
        stop_event, thread = manager.start_heartbeat(write_interval=0.1)

        # Wait a bit for heartbeat to write
        time.sleep(0.3)

        # Verify file was created
        self.assertTrue(manager.hb_file.exists())

        # Stop the heartbeat
        stop_event.set()
        thread.join(timeout=1.0)

        # Verify thread stopped
        self.assertFalse(thread.is_alive())

    def test_start_heartbeat_writes_periodically(self):
        """Test that heartbeat writes periodically."""
        manager = HeartBeatManager(self.temp_dir)
        stop_event, thread = manager.start_heartbeat(write_interval=0.1)

        # Wait for multiple writes
        time.sleep(0.35)

        # Verify file exists and has content
        self.assertTrue(manager.hb_file.exists())
        content = manager.hb_file.read_text()
        self.assertIsNotNone(content)
        self.assertNotEqual(content, '')

        # Stop the heartbeat
        stop_event.set()
        thread.join(timeout=1.0)

    def test_start_heartbeat_daemon_thread(self):
        """Test that heartbeat thread is daemon."""
        manager = HeartBeatManager(self.temp_dir)
        stop_event, thread = manager.start_heartbeat()

        self.assertTrue(thread.daemon)

        stop_event.set()
        thread.join(timeout=1.0)

    def test_last_heartbeat_with_invalid_isoformat(self):
        """Test last_heartbeat handles invalid isoformat gracefully."""
        manager = HeartBeatManager(self.temp_dir)
        manager.hb_file.parent.mkdir(parents=True, exist_ok=True)
        manager.hb_file.write_text('invalid_format')

        # Should fall back to file mtime
        result = manager.last_heartbeat()
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

    def test_heartbeat_thread_stops_on_event(self):
        """Test that heartbeat thread stops when event is set."""
        manager = HeartBeatManager(self.temp_dir)
        stop_event, thread = manager.start_heartbeat(write_interval=0.1)

        # Let it run briefly
        time.sleep(0.2)

        # Stop it
        stop_event.set()

        # Wait for thread to finish
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()
