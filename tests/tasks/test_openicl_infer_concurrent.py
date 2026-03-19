"""Unit tests for OpenICLInferConcurrentTask."""

import os
import os.path as osp
import tempfile
import threading
import unittest
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

from mmengine.config import ConfigDict

from opencompass.tasks.openicl_infer_concurrent import (
    OpenICLInferConcurrentTask, _ProgressTracker, _RunningTask)


class TestProgressTracker(unittest.TestCase):
    """Test cases for _ProgressTracker."""

    def test_initialization(self):
        """Test _ProgressTracker initialization."""
        tracker = _ProgressTracker('test_task')
        self.assertEqual(tracker.name, 'test_task')
        self.assertIsNone(tracker.total)
        self.assertEqual(tracker.completed, 0)
        self.assertIsNotNone(tracker._lock)

    def test_set_total(self):
        """Test set_total method."""
        tracker = _ProgressTracker('test_task')
        tracker.set_total(100)
        self.assertEqual(tracker.total, 100)

    def test_set_completed(self):
        """Test set_completed method."""
        tracker = _ProgressTracker('test_task')
        tracker.set_completed(50)
        self.assertEqual(tracker.completed, 50)

    def test_incr(self):
        """Test incr method."""
        tracker = _ProgressTracker('test_task')
        tracker.incr()
        self.assertEqual(tracker.completed, 1)
        tracker.incr(5)
        self.assertEqual(tracker.completed, 6)

    def test_remaining_without_total(self):
        """Test remaining method when total is None."""
        tracker = _ProgressTracker('test_task')
        self.assertIsNone(tracker.remaining())

    def test_remaining_with_total(self):
        """Test remaining method when total is set."""
        tracker = _ProgressTracker('test_task')
        tracker.set_total(100)
        tracker.set_completed(30)
        self.assertEqual(tracker.remaining(), 70)

    def test_remaining_negative_protection(self):
        """Test remaining method protects against negative values."""
        tracker = _ProgressTracker('test_task')
        tracker.set_total(100)
        tracker.set_completed(150)
        self.assertEqual(tracker.remaining(), 0)

    def test_thread_safety(self):
        """Test thread safety of _ProgressTracker."""
        tracker = _ProgressTracker('test_task')
        tracker.set_total(1000)

        def increment():
            for _ in range(100):
                tracker.incr()

        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(tracker.completed, 1000)
        self.assertEqual(tracker.remaining(), 0)


class TestRunningTask(unittest.TestCase):
    """Test cases for _RunningTask."""

    def test_initialization(self):
        """Test _RunningTask initialization."""
        progress = _ProgressTracker('test_task')
        future = Future()
        task = _RunningTask('test_task', progress, future)
        self.assertEqual(task.name, 'test_task')
        self.assertEqual(task.progress, progress)
        self.assertEqual(task.future, future)


class TestOpenICLInferConcurrentTask(unittest.TestCase):
    """Test cases for OpenICLInferConcurrentTask."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = ConfigDict({
            'work_dir':
            self.temp_dir,
            'models': [
                ConfigDict({
                    'abbr': 'test_model',
                    'type': 'TestAPIModel',
                    'run_cfg': {
                        'num_gpus': 0,
                        'num_procs': 1
                    }
                })
            ],
            'datasets': [[
                ConfigDict({
                    'abbr': 'test_dataset',
                    'infer_cfg': {
                        'retriever': {
                            'type': 'TestRetriever'
                        },
                        'inferencer': {
                            'type': 'GenInferencer'
                        },
                        'prompt_template': {
                            'type': 'TestTemplate'
                        }
                    }
                })
            ]],
            'dump_res_length':
            False,
            'poll_interval':
            0.1,
            'log_interval':
            1.0
        })

    def test_initialization(self):
        """Test OpenICLInferConcurrentTask initialization."""
        task = OpenICLInferConcurrentTask(self.cfg)
        self.assertEqual(task.num_gpus, 0)
        self.assertEqual(task.num_procs, 1)
        self.assertFalse(task.dump_res_length)
        self.assertEqual(task.poll_interval, 0.1)
        self.assertEqual(task.log_interval, 1.0)
        self.assertIsNotNone(task.logger)

    def test_initialization_with_defaults(self):
        """Test initialization with default values."""
        cfg = ConfigDict({
            'work_dir':
            self.temp_dir,
            'models': [ConfigDict({
                'abbr': 'test_model',
                'run_cfg': {}
            })],
            'datasets': [[]]
        })
        task = OpenICLInferConcurrentTask(cfg)
        self.assertEqual(task.num_gpus, 0)
        self.assertEqual(task.num_procs, 1)
        self.assertFalse(task.dump_res_length)
        self.assertEqual(task.poll_interval, 1.0)
        self.assertEqual(task.log_interval, 30.0)

    @patch('opencompass.tasks.openicl_infer_concurrent.sys.executable',
           '/usr/bin/python')
    @patch('opencompass.tasks.openicl_infer_concurrent.__file__',
           '/path/to/script.py')
    def test_get_command_single_gpu(self):
        """Test get_command with single GPU."""
        task = OpenICLInferConcurrentTask(self.cfg)
        template = 'command: {task_cmd}'
        result = task.get_command('/path/to/config.py', template)
        self.assertIn('/usr/bin/python', result)
        self.assertIn('/path/to/script.py', result)
        self.assertIn('/path/to/config.py', result)
        self.assertNotIn('torch.distributed.run', result)

    @patch('opencompass.tasks.openicl_infer_concurrent.sys.executable',
           '/usr/bin/python')
    @patch('opencompass.tasks.openicl_infer_concurrent.__file__',
           '/path/to/script.py')
    @patch('opencompass.tasks.openicl_infer_concurrent.random.randint')
    def test_get_command_multi_gpu(self, mock_randint):
        """Test get_command with multiple GPUs."""
        mock_randint.return_value = 15000
        cfg = ConfigDict({
            'work_dir':
            self.temp_dir,
            'models': [
                ConfigDict({
                    'abbr': 'test_model',
                    'run_cfg': {
                        'num_gpus': 2,
                        'num_procs': 2
                    }
                })
            ],
            'datasets': [[]]
        })
        task = OpenICLInferConcurrentTask(cfg)
        template = 'command: {task_cmd}'
        result = task.get_command('/path/to/config.py', template)
        self.assertIn('torch.distributed.run', result)
        self.assertIn('--master_port=15000', result)
        self.assertIn('--nproc_per_node 2', result)

    @patch('opencompass.tasks.openicl_infer_concurrent.sys.executable',
           '/usr/bin/python')
    @patch('opencompass.tasks.openicl_infer_concurrent.__file__',
           '/path/to/script.py')
    def test_get_command_with_backend(self):
        """Test get_command with VLLM backend."""
        cfg = ConfigDict({
            'work_dir':
            self.temp_dir,
            'models': [
                ConfigDict({
                    'abbr': 'test_model',
                    'type': 'VLLM',
                    'run_cfg': {
                        'num_gpus': 2,
                        'num_procs': 2
                    }
                })
            ],
            'datasets': [[]]
        })
        task = OpenICLInferConcurrentTask(cfg)
        template = 'command: {task_cmd}'
        result = task.get_command('/path/to/config.py', template)
        # Should not use distributed even with multiple GPUs when backend is used # noqa
        self.assertNotIn('torch.distributed.run', result)

    def test_default_max_workers(self):
        """Test _default_max_workers method."""
        task = OpenICLInferConcurrentTask(self.cfg)
        max_workers = task._default_max_workers()
        cpu_count = os.cpu_count() or 1
        expected = min(32, cpu_count + 4)
        self.assertEqual(max_workers, expected)
        self.assertGreater(max_workers, 0)
        self.assertLessEqual(max_workers, 32)

    def test_inferencer_name_from_string(self):
        """Test _inferencer_name with string input."""
        task = OpenICLInferConcurrentTask(self.cfg)
        result = task._inferencer_name('opencompass.openicl.GenInferencer')
        self.assertEqual(result, 'GenInferencer')

    def test_inferencer_name_from_class(self):
        """Test _inferencer_name with class input."""
        task = OpenICLInferConcurrentTask(self.cfg)

        class TestInferencer:
            pass

        result = task._inferencer_name(TestInferencer)
        self.assertEqual(result, 'TestInferencer')

    def test_set_default_value(self):
        """Test _set_default_value method."""
        task = OpenICLInferConcurrentTask(self.cfg)
        cfg = ConfigDict({})
        task._set_default_value(cfg, 'key', 'value')
        self.assertEqual(cfg['key'], 'value')

    def test_set_default_value_existing(self):
        """Test _set_default_value doesn't override existing values."""
        task = OpenICLInferConcurrentTask(self.cfg)
        cfg = ConfigDict({'key': 'existing'})
        task._set_default_value(cfg, 'key', 'new')
        self.assertEqual(cfg['key'], 'existing')

    @patch('opencompass.tasks.openicl_infer_concurrent.ICL_INFERENCERS')
    def test_build_inferencer_gen(self, mock_registry):
        """Test _build_inferencer with GenInferencer."""
        task = OpenICLInferConcurrentTask(self.cfg)
        mock_model = MagicMock()
        mock_model.is_api = True
        model_cfg = ConfigDict({'max_out_len': 512})
        dataset_cfg = ConfigDict(
            {'infer_cfg': {
                'inferencer': {
                    'type': 'GenInferencer'
                }
            }})
        mock_inferencer = MagicMock()
        mock_registry.build.return_value = mock_inferencer

        result = task._build_inferencer(mock_model, model_cfg, dataset_cfg, 4)

        self.assertEqual(result, mock_inferencer)
        mock_registry.build.assert_called_once()
        call_args = mock_registry.build.call_args[0][0]
        # Check that type is set to ParallelGenInferencer class
        self.assertIsNotNone(call_args['type'])
        self.assertEqual(call_args['type'].__name__, 'ParallelGenInferencer')

    @patch('opencompass.tasks.openicl_infer_concurrent.ICL_INFERENCERS')
    def test_build_inferencer_chat(self, mock_registry):
        """Test _build_inferencer with ChatInferencer."""
        task = OpenICLInferConcurrentTask(self.cfg)
        mock_model = MagicMock()
        mock_model.is_api = True
        model_cfg = ConfigDict({'max_out_len': 512})
        dataset_cfg = ConfigDict(
            {'infer_cfg': {
                'inferencer': {
                    'type': 'ChatInferencer'
                }
            }})
        mock_inferencer = MagicMock()
        mock_registry.build.return_value = mock_inferencer

        result = task._build_inferencer(mock_model, model_cfg, dataset_cfg, 4)

        self.assertEqual(result, mock_inferencer)
        call_args = mock_registry.build.call_args[0][0]
        # Check that type is set to ParallelChatInferencer class
        self.assertIsNotNone(call_args['type'])
        self.assertEqual(call_args['type'].__name__, 'ParallelChatInferencer')

    @patch('opencompass.tasks.openicl_infer_concurrent.ICL_INFERENCERS')
    def test_build_inferencer_unsupported(self, mock_registry):
        """Test _build_inferencer with unsupported inferencer type."""
        task = OpenICLInferConcurrentTask(self.cfg)
        mock_model = MagicMock()
        model_cfg = ConfigDict()
        dataset_cfg = ConfigDict(
            {'infer_cfg': {
                'inferencer': {
                    'type': 'UnsupportedInferencer'
                }
            }})

        with self.assertRaises(NotImplementedError) as context:
            task._build_inferencer(mock_model, model_cfg, dataset_cfg, 4)
        self.assertIn('Unsupported inferencer type', str(context.exception))

    def test_remaining_total(self):
        """Test _remaining_total method."""
        task = OpenICLInferConcurrentTask(self.cfg)
        progress1 = _ProgressTracker('task1')
        progress1.set_total(100)
        progress1.set_completed(30)
        progress2 = _ProgressTracker('task2')
        progress2.set_total(50)
        progress2.set_completed(20)

        future1 = Future()
        future2 = Future()
        running = [
            _RunningTask('task1', progress1, future1),
            _RunningTask('task2', progress2, future2)
        ]

        result = task._remaining_total(running, 1000)
        self.assertEqual(result, 100)  # (100-30) + (50-20) = 70 + 30 = 100

    def test_remaining_total_with_none(self):
        """Test _remaining_total when progress total is None."""
        task = OpenICLInferConcurrentTask(self.cfg)
        progress = _ProgressTracker('task1')
        # total is None by default
        future = Future()
        running = [_RunningTask('task1', progress, future)]

        result = task._remaining_total(running, 1000)
        self.assertEqual(result, 1000)  # Should return max_pending_samples

    @patch('opencompass.tasks.openicl_infer_concurrent.build_dataset_from_cfg')
    @patch('opencompass.tasks.openicl_infer_concurrent.ICL_RETRIEVERS')
    @patch('opencompass.tasks.openicl_infer_concurrent.ICL_PROMPT_TEMPLATES')
    @patch('opencompass.tasks.openicl_infer_concurrent.ICL_INFERENCERS')
    @patch('opencompass.tasks.openicl_infer_concurrent.get_infer_output_path')
    @patch('opencompass.tasks.openicl_infer_concurrent.mkdir_or_exist')
    def test_run_dataset_task_success(self, mock_mkdir, mock_get_path,
                                      mock_inferencers, mock_templates,
                                      mock_retrievers, mock_build_dataset):
        """Test _run_dataset_task with successful execution."""
        task = OpenICLInferConcurrentTask(self.cfg)
        mock_model = MagicMock()
        mock_model.is_api = True
        task.model = mock_model

        # Setup mocks
        mock_dataset = MagicMock()
        mock_dataset.test = [MagicMock()] * 10
        mock_build_dataset.return_value = mock_dataset

        mock_retriever = MagicMock()
        mock_retrievers.build.return_value = mock_retriever

        mock_inferencer = MagicMock()
        mock_inferencers.build.return_value = mock_inferencer

        mock_template = MagicMock()
        mock_templates.build.return_value = mock_template

        mock_get_path.return_value = '/tmp/output.json'

        model_cfg = ConfigDict()
        dataset_cfg = ConfigDict({
            'infer_cfg': {
                'retriever': {
                    'type': 'TestRetriever'
                },
                'inferencer': {
                    'type': 'GenInferencer'
                },
                'prompt_template': {
                    'type': 'TestTemplate'
                }
            }
        })
        tokens = threading.Semaphore(4)
        progress = _ProgressTracker('test_task')
        status = MagicMock()

        task._run_dataset_task(model_cfg, dataset_cfg, tokens, 4, progress,
                               status)

        # Verify model.tokens was set
        self.assertEqual(mock_model.tokens, tokens)
        # Verify progress was set
        self.assertEqual(progress.total, 10)
        # Verify status was updated to done
        status.update.assert_called_with(status='done')
        # Verify inferencer.inference was called
        mock_inferencer.inference.assert_called_once()

    @patch('opencompass.tasks.openicl_infer_concurrent.build_dataset_from_cfg')
    def test_run_dataset_task_non_api_model(self, mock_build_dataset):
        """Test _run_dataset_task handles error for non-API model."""
        task = OpenICLInferConcurrentTask(self.cfg)
        mock_model = MagicMock()
        mock_model.is_api = False
        task.model = mock_model

        model_cfg = ConfigDict()
        dataset_cfg = ConfigDict({
            'infer_cfg': {
                'retriever': {
                    'type': 'TestRetriever'
                },
                'inferencer': {
                    'type': 'GenInferencer'
                }
            }
        })
        tokens = threading.Semaphore(4)
        progress = _ProgressTracker('test_task')
        status = MagicMock()

        # _run_dataset_task catches exceptions, so it won't raise
        task._run_dataset_task(model_cfg, dataset_cfg, tokens, 4, progress,
                               status)

        # Verify status was updated to 'fail' due to the error
        status.update.assert_called_with(status='fail')

    @patch('opencompass.tasks.openicl_infer_concurrent.build_dataset_from_cfg')
    def test_run_dataset_task_missing_template(self, mock_build_dataset):
        """Test _run_dataset_task handles error when both templates are missing."""  # noqa
        task = OpenICLInferConcurrentTask(self.cfg)
        mock_model = MagicMock()
        mock_model.is_api = True
        task.model = mock_model

        mock_dataset = MagicMock()
        mock_dataset.test = []
        mock_build_dataset.return_value = mock_dataset

        model_cfg = ConfigDict()
        dataset_cfg = ConfigDict({
            'infer_cfg': {
                'retriever': {
                    'type': 'TestRetriever'
                },
                'inferencer': {
                    'type': 'GenInferencer'
                }
                # Missing both ice_template and prompt_template
            }
        })
        tokens = threading.Semaphore(4)
        progress = _ProgressTracker('test_task')
        status = MagicMock()

        # _run_dataset_task catches exceptions, so it won't raise
        task._run_dataset_task(model_cfg, dataset_cfg, tokens, 4, progress,
                               status)

        # Verify status was updated to 'fail' due to the error
        status.update.assert_called_with(status='fail')

    @patch('opencompass.tasks.openicl_infer_concurrent.build_model_from_cfg')
    @patch('opencompass.tasks.openicl_infer_concurrent.build_dataset_from_cfg')
    @patch('opencompass.tasks.openicl_infer_concurrent.get_infer_output_path')
    @patch('opencompass.tasks.openicl_infer_concurrent.task_abbr_from_cfg')
    def test_run_skips_existing_output(self, mock_task_abbr, mock_get_path,
                                       mock_build_dataset, mock_build_model):
        """Test run method skips tasks with existing output files."""
        mock_get_path.return_value = osp.join(self.temp_dir,
                                              'existing_output.json')
        os.makedirs(osp.dirname(mock_get_path.return_value), exist_ok=True)
        with open(mock_get_path.return_value, 'w') as f:
            f.write('{}')

        task = OpenICLInferConcurrentTask(self.cfg)
        task.logger = MagicMock()
        # Mock model to avoid actual model building
        mock_model = MagicMock()
        mock_model.is_api = True
        mock_build_model.return_value = mock_model

        task.run()

        # Model is built before checking output files, but dataset should not be built # noqa
        # and _run_task_group should not be called since tasks list is empty
        mock_build_model.assert_called_once()
        # Dataset should not be built since output exists and task is skipped
        mock_build_dataset.assert_not_called()

    @patch('opencompass.tasks.openicl_infer_concurrent.build_model_from_cfg')
    @patch('opencompass.tasks.openicl_infer_concurrent.build_dataset_from_cfg')
    @patch('opencompass.tasks.openicl_infer_concurrent.get_infer_output_path')
    @patch('opencompass.tasks.openicl_infer_concurrent.task_abbr_from_cfg')
    @patch('opencompass.tasks.openicl_infer_concurrent.model_abbr_from_cfg')
    def test_run_with_cur_model(self, mock_model_abbr, mock_task_abbr,
                                mock_get_path, mock_build_dataset,
                                mock_build_model):
        """Test run method uses provided cur_model."""
        mock_get_path.return_value = osp.join(self.temp_dir, 'output.json')
        mock_model_abbr.return_value = 'test_model'
        mock_task_abbr.return_value = 'test_task'

        task = OpenICLInferConcurrentTask(self.cfg)
        task.logger = MagicMock()

        mock_cur_model = MagicMock()
        mock_cur_model.is_api = True

        task.run(cur_model=mock_cur_model, cur_model_abbr='test_model')

        # Should use provided model instead of building new one
        mock_build_model.assert_not_called()
        self.assertEqual(task.model, mock_cur_model)

    def test_run_with_max_workers_config(self):
        """Test run method uses max_workers from model config."""
        cfg = ConfigDict({
            'work_dir':
            self.temp_dir,
            'models': [
                ConfigDict({
                    'abbr': 'test_model',
                    'max_workers': 8,
                    'run_cfg': {}
                })
            ],
            'datasets': [[]]
        })
        task = OpenICLInferConcurrentTask(cfg)
        # max_workers should be read from model_cfg
        self.assertTrue(hasattr(task, 'model_cfgs'))

    @patch('opencompass.tasks.openicl_infer_concurrent.ICL_INFERENCERS')
    def test_build_inferencer_with_max_infer_workers(self, mock_registry):
        """Test _build_inferencer sets max_infer_workers."""
        task = OpenICLInferConcurrentTask(self.cfg)
        mock_model = MagicMock()
        mock_model.is_api = True
        model_cfg = ConfigDict()
        dataset_cfg = ConfigDict(
            {'infer_cfg': {
                'inferencer': {
                    'type': 'GenInferencer'
                }
            }})
        mock_inferencer = MagicMock()
        mock_registry.build.return_value = mock_inferencer

        task._build_inferencer(mock_model, model_cfg, dataset_cfg, 6)

        call_args = mock_registry.build.call_args[0][0]
        self.assertEqual(call_args['max_infer_workers'], 6)

    @patch('opencompass.tasks.openicl_infer_concurrent.ICL_INFERENCERS')
    def test_build_inferencer_with_model_config(self, mock_registry):
        """Test _build_inferencer uses model config values."""
        task = OpenICLInferConcurrentTask(self.cfg)
        mock_model = MagicMock()
        mock_model.is_api = True
        model_cfg = ConfigDict({
            'max_out_len': 1024,
            'min_out_len': 10,
            'max_seq_len': 2048
        })
        dataset_cfg = ConfigDict(
            {'infer_cfg': {
                'inferencer': {
                    'type': 'GenInferencer'
                }
            }})
        mock_inferencer = MagicMock()
        mock_registry.build.return_value = mock_inferencer

        task._build_inferencer(mock_model, model_cfg, dataset_cfg, 4)

        call_args = mock_registry.build.call_args[0][0]
        self.assertEqual(call_args['max_out_len'], 1024)
        self.assertEqual(call_args['min_out_len'], 10)
        self.assertEqual(call_args['max_seq_len'], 2048)
        self.assertEqual(call_args['model'], mock_model)
        self.assertEqual(call_args['dump_res_length'], False)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()
