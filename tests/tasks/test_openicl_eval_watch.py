"""Unit tests for OpenICLEvalWatchTask."""

import os
import os.path as osp
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from mmengine.config import ConfigDict

from opencompass.tasks.openicl_eval_watch import OpenICLEvalWatchTask


class TestOpenICLEvalWatchTask(unittest.TestCase):
    """Test cases for OpenICLEvalWatchTask."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = ConfigDict({
            'work_dir':
            self.temp_dir,
            'models': [ConfigDict({'abbr': 'test_model'})],
            'datasets': [[
                ConfigDict({
                    'abbr': 'test_dataset',
                    'reader_cfg': {
                        'output_column': 'answer'
                    },
                    'eval_cfg': {}
                })
            ]],
            'eval': {
                'runner': {
                    'task': {
                        'watch_interval': 1.0,
                        'heartbeat_timeout': 30.0,
                        'log_interval': 10.0
                    }
                }
            }
        })

    def test_initialization(self):
        """Test OpenICLEvalWatchTask initialization."""
        task = OpenICLEvalWatchTask(self.cfg)
        self.assertEqual(task.watch_interval, 1.0)
        self.assertEqual(task.heartbeat_timeout, 30.0)
        self.assertEqual(task.log_interval, 10.0)
        self.assertIsNotNone(task.heartbeat)

    def test_initialization_with_defaults(self):
        """Test initialization with default values."""
        cfg = ConfigDict({
            'work_dir':
            self.temp_dir,
            'models': [ConfigDict({'abbr': 'test_model'})],
            'datasets': [[
                ConfigDict({
                    'abbr': 'test_dataset',
                    'reader_cfg': {
                        'output_column': 'answer'
                    },
                    'eval_cfg': {}
                })
            ]],
            'eval': {
                'runner': {
                    'task': {}
                }
            }
        })
        task = OpenICLEvalWatchTask(cfg)
        self.assertEqual(task.watch_interval, 5.0)
        self.assertEqual(task.heartbeat_timeout, 60.0)
        self.assertEqual(task.log_interval, 30.0)

    @patch('opencompass.tasks.openicl_eval_watch.sys.executable',
           '/usr/bin/python')
    @patch('opencompass.tasks.openicl_eval_watch.__file__',
           '/path/to/script.py')
    def test_get_command_single_gpu(self):
        """Test get_command with single GPU."""
        task = OpenICLEvalWatchTask(self.cfg)
        template = 'command: {task_cmd}'
        result = task.get_command('/path/to/config.py', template)
        self.assertIn('/usr/bin/python', result)
        self.assertIn('/path/to/script.py', result)
        self.assertIn('/path/to/config.py', result)
        self.assertNotIn('torch.distributed.run', result)

    @patch('opencompass.tasks.openicl_eval_watch.sys.executable',
           '/usr/bin/python')
    @patch('opencompass.tasks.openicl_eval_watch.__file__',
           '/path/to/script.py')
    @patch('opencompass.tasks.openicl_eval_watch.random.randint')
    def test_get_command_multi_gpu(self, mock_randint):
        """Test get_command with multiple GPUs."""
        mock_randint.return_value = 15000
        cfg = ConfigDict({
            'work_dir':
            self.temp_dir,
            'models': [ConfigDict({'abbr': 'test_model'})],
            'datasets': [[
                ConfigDict({
                    'abbr': 'test_dataset',
                    'reader_cfg': {
                        'output_column': 'answer'
                    },
                    'eval_cfg': {}
                })
            ]],
            'eval': {
                'runner': {
                    'task': {}
                }
            }
        })
        task = OpenICLEvalWatchTask(cfg)
        task.num_gpus = 2
        task.num_procs = 2
        template = 'command: {task_cmd}'
        result = task.get_command('/path/to/config.py', template)
        self.assertIn('torch.distributed.run', result)
        self.assertIn('--master_port=15000', result)
        self.assertIn('--nproc_per_node 2', result)

    @patch('opencompass.tasks.openicl_eval_watch.get_infer_output_path')
    def test_run_skips_finished_tasks(self, mock_get_path):
        """Test run method skips finished tasks."""
        mock_get_path.return_value = osp.join(self.temp_dir,
                                              'existing_result.json')
        os.makedirs(osp.dirname(mock_get_path.return_value), exist_ok=True)
        with open(mock_get_path.return_value, 'w') as f:
            f.write('{}')

        task = OpenICLEvalWatchTask(self.cfg)
        task.logger = MagicMock()
        task._score = MagicMock()

        task.run()

        # Should not call _score since all tasks are finished
        task._score.assert_not_called()

    @patch('opencompass.tasks.openicl_eval_watch.get_infer_output_path')
    @patch('opencompass.tasks.openicl_eval_watch.InferStatusManager')
    def test_is_ready_all_done(self, mock_status_manager, mock_get_path):
        """Test _is_ready when all statuses are done."""
        mock_get_path.return_value = osp.join(self.temp_dir, 'result.json')

        mock_status = MagicMock()
        mock_status.get_task_status.return_value = {
            'task1': {
                'status': 'done'
            },
            'task2': {
                'status': 'done'
            }
        }
        mock_status_manager.return_value = mock_status

        task = OpenICLEvalWatchTask(self.cfg)
        task.logger = MagicMock()

        model_cfg = ConfigDict({'abbr': 'test_model'})
        dataset_cfg = ConfigDict({'abbr': 'test_dataset'})
        status_index = {('test_model', 'test_dataset'): mock_status}

        result = task._is_ready(model_cfg, dataset_cfg, status_index)
        self.assertTrue(result)

    @patch('opencompass.tasks.openicl_eval_watch.get_infer_output_path')
    @patch('opencompass.tasks.openicl_eval_watch.InferStatusManager')
    def test_is_ready_not_all_done(self, mock_status_manager, mock_get_path):
        """Test _is_ready when not all statuses are done."""
        mock_get_path.return_value = osp.join(self.temp_dir, 'result.json')

        mock_status = MagicMock()
        mock_status.get_task_status.return_value = {
            'task1': {
                'status': 'done'
            },
            'task2': {
                'status': 'running'
            }
        }
        mock_status_manager.return_value = mock_status

        task = OpenICLEvalWatchTask(self.cfg)
        task.logger = MagicMock()

        model_cfg = ConfigDict({'abbr': 'test_model'})
        dataset_cfg = ConfigDict({'abbr': 'test_dataset'})
        status_index = {('test_model', 'test_dataset'): mock_status}

        result = task._is_ready(model_cfg, dataset_cfg, status_index)
        self.assertFalse(result)

    @patch('opencompass.tasks.openicl_eval_watch.get_infer_output_path')
    @patch('opencompass.tasks.openicl_eval_watch.InferStatusManager')
    def test_is_ready_no_status(self, mock_status_manager, mock_get_path):
        """Test _is_ready when status is empty."""
        mock_get_path.return_value = osp.join(self.temp_dir, 'result.json')

        mock_status = MagicMock()
        mock_status.get_task_status.return_value = {}
        mock_status_manager.return_value = mock_status

        task = OpenICLEvalWatchTask(self.cfg)
        task.logger = MagicMock()

        model_cfg = ConfigDict({'abbr': 'test_model'})
        dataset_cfg = ConfigDict({'abbr': 'test_dataset'})
        status_index = {('test_model', 'test_dataset'): mock_status}

        result = task._is_ready(model_cfg, dataset_cfg, status_index)
        self.assertFalse(result)

    @patch('opencompass.tasks.openicl_eval_watch.get_infer_output_path')
    @patch('opencompass.tasks.openicl_eval_watch.InferStatusManager')
    @patch('opencompass.tasks.openicl_eval_watch.time.sleep')
    def test_run_with_ready_tasks(self, mock_sleep, mock_status_manager,
                                  mock_get_path):
        """Test run method processes ready tasks."""
        mock_get_path.return_value = osp.join(self.temp_dir, 'result.json')

        mock_status = MagicMock()
        mock_status.get_task_status.return_value = {
            'task1': {
                'status': 'done'
            }
        }
        mock_status_manager.return_value = mock_status

        task = OpenICLEvalWatchTask(self.cfg)
        task.logger = MagicMock()
        task._score = MagicMock()

        # Make _is_ready return True immediately
        task._is_ready = MagicMock(return_value=True)

        # Mock time.sleep to break the loop after first iteration
        call_count = [0]

        def side_effect(*args):
            call_count[0] += 1
            if call_count[0] > 1:
                # Break the loop by making pending empty
                task.model_cfgs = []

        mock_sleep.side_effect = side_effect

        task.run()

        # Should call _score for ready tasks
        task._score.assert_called()

    @patch('opencompass.tasks.openicl_eval_watch.get_infer_output_path')
    @patch('opencompass.tasks.openicl_eval_watch.InferStatusManager')
    @patch('opencompass.tasks.openicl_eval_watch.time.sleep')
    @patch('opencompass.tasks.openicl_eval_watch.time.time')
    def test_run_heartbeat_timeout(self, mock_time, mock_sleep,
                                   mock_status_manager, mock_get_path):
        """Test run method handles heartbeat timeout."""
        mock_get_path.return_value = osp.join(self.temp_dir, 'result.json')

        mock_status = MagicMock()
        mock_status.get_task_status.return_value = {
            'task1': {
                'status': 'running'
            }
        }
        mock_status_manager.return_value = mock_status

        task = OpenICLEvalWatchTask(self.cfg)
        task.logger = MagicMock()
        task._score = MagicMock()
        task._is_ready = MagicMock(return_value=False)

        # Mock heartbeat to return timeout
        task.heartbeat.last_heartbeat = MagicMock(return_value=100.0)

        # Mock time to control loop
        mock_time.return_value = 0
        call_count = [0]

        def sleep_side_effect(*args):
            call_count[0] += 1
            if call_count[0] > 1:
                task.model_cfgs = []  # Break loop

        mock_sleep.side_effect = sleep_side_effect

        task.run()

        # Should log warning about timeout
        task.logger.warning.assert_called()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()
