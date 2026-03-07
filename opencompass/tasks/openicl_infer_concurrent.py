import argparse
import os
import os.path as osp
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

import opencompass.datasets  # noqa: F401
from opencompass.openicl.icl_inferencer import (ParallelChatInferencer,
                                                ParallelGenInferencer)
from opencompass.registry import (ICL_INFERENCERS, ICL_PROMPT_TEMPLATES,
                                  ICL_RETRIEVERS, TASKS)
from opencompass.tasks.base import BaseTask
from opencompass.utils import (InferStatusManager, build_dataset_from_cfg,
                               build_model_from_cfg, get_infer_output_path,
                               get_logger, model_abbr_from_cfg,
                               task_abbr_from_cfg)


@dataclass
class _ProgressTracker:
    name: str
    total: Optional[int] = None
    completed: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set_total(self, total: int) -> None:
        with self._lock:
            self.total = total

    def set_completed(self, completed: int) -> None:
        with self._lock:
            self.completed = completed

    def incr(self, count: int = 1) -> None:
        with self._lock:
            self.completed += count

    def remaining(self) -> Optional[int]:
        with self._lock:
            if self.total is None:
                return None
            return max(self.total - self.completed, 0)


@dataclass
class _RunningTask:
    name: str
    progress: _ProgressTracker
    future: Any


@TASKS.register_module()
class OpenICLInferTaskConcurrent(BaseTask):
    """Concurrent OpenICL Inference Task for API models."""

    name_prefix = 'OpenICLInferConcurrent'
    log_subdir = 'logs/infer'
    output_subdir = 'predictions'

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        run_cfg = self.model_cfgs[0].get('run_cfg', {})
        self.num_gpus = run_cfg.get('num_gpus', 0)
        self.num_procs = run_cfg.get('num_procs', 1)
        self.logger = get_logger()
        self.dump_res_length = cfg.get('dump_res_length', False)
        self.poll_interval = cfg.get('poll_interval', 1.0)
        self.log_interval = cfg.get('log_interval', 10.0)

    def get_command(self, cfg_path, template):
        sys.path.append(os.getcwd())
        script_path = __file__
        backend_keys = ['VLLM', 'Lmdeploy']
        use_backend = any(
            key in str(self.model_cfgs[0].get('type', ''))
            or key in str(self.model_cfgs[0].get('llm', {}).get('type', ''))
            for key in backend_keys)
        python = sys.executable
        if self.num_gpus > 1 and not use_backend:
            port = random.randint(12000, 32000)
            command = (
                f'{python} -m torch.distributed.run --master_port={port} '
                f'--nproc_per_node {self.num_procs} '
                f'{script_path} {cfg_path}')
        else:
            command = f'{python} {script_path} {cfg_path}'

        return template.format(task_cmd=command)

    def run(self, cur_model=None, cur_model_abbr=None):
        self.logger.info(f'Task {task_abbr_from_cfg(self.cfg)}')
        for model_cfg, dataset_cfgs in zip(self.model_cfgs, self.dataset_cfgs):

            self.model_cfg = model_cfg
            if cur_model and cur_model_abbr == model_abbr_from_cfg(model_cfg):
                self.model = cur_model
            else:
                self.model = build_model_from_cfg(model_cfg)

            tasks = []
            for dataset_cfg in dataset_cfgs:
                out_path = get_infer_output_path(
                    model_cfg, dataset_cfg,
                    osp.join(self.work_dir, 'predictions'))
                if osp.exists(out_path):
                    continue
                tasks.append(dataset_cfg)

            if not tasks:
                continue

            # Default concurrent controls from run_cfg or model config.
            max_workers = model_cfg.get('max_workers',
                                        self._default_max_workers())
            max_workers = max(1, int(max_workers))

            tokens = threading.Semaphore(max_workers)
            self.logger.info(
                f'Concurrent infer settings: max_workers={max_workers}.')

            self._run_task_group(model_cfg, tasks, tokens, max_workers)

    def _default_max_workers(self) -> int:
        cpu_count = os.cpu_count() or 1
        return min(32, cpu_count + 4)

    def _inferencer_name(self, inferencer_type) -> str:
        if isinstance(inferencer_type, str):
            return inferencer_type.split('.')[-1]
        return inferencer_type.__name__

    def _set_default_value(self, cfg: ConfigDict, key: str, value: Any):
        if key not in cfg:
            cfg[key] = value

    def _build_inferencer(self, model, model_cfg, dataset_cfg, max_workers):
        infer_cfg = dataset_cfg['infer_cfg']
        inferencer_cfg = infer_cfg['inferencer'].copy()
        inferencer_type = self._inferencer_name(inferencer_cfg.get('type'))
        if inferencer_type == 'GenInferencer':
            inferencer_cfg['type'] = ParallelGenInferencer
        elif inferencer_type == 'ChatInferencer':
            inferencer_cfg['type'] = ParallelChatInferencer

        if inferencer_type not in ('ChatInferencer', 'GenInferencer',
                                   'ParallelChatInferencer',
                                   'ParallelGenInferencer'):
            raise NotImplementedError(
                f'Unsupported inferencer type `{inferencer_type}` '
                'for OpenICLInferTaskConcurrent')

        inferencer_cfg.setdefault('max_infer_workers', max_workers)

        inferencer_cfg['model'] = model
        self._set_default_value(inferencer_cfg, 'max_out_len',
                                model_cfg.get('max_out_len', None))
        self._set_default_value(inferencer_cfg, 'min_out_len',
                                model_cfg.get('min_out_len', None))
        inferencer_cfg['max_seq_len'] = model_cfg.get('max_seq_len')
        inferencer_cfg['dump_res_length'] = self.dump_res_length
        return ICL_INFERENCERS.build(inferencer_cfg)

    def _run_dataset_task(self, model_cfg, dataset_cfg, tokens, max_workers,
                          progress: _ProgressTracker,
                          status: InferStatusManager):
        try:
            model = self.model
            if model.is_api:
                model.tokens = tokens
            else:
                raise RuntimeError(
                    'OpenICLInferTaskConcurrent only supports API models, '
                    f'got {type(model)} instead.')

            dataset = build_dataset_from_cfg(dataset_cfg)
            progress.set_total(len(dataset.test))

            infer_cfg = dataset_cfg['infer_cfg']
            assert hasattr(infer_cfg, 'ice_template') or hasattr(
                infer_cfg, 'prompt_template'), (
                    'Both ice_template and prompt_template cannot be None '
                    'simultaneously.')

            retriever_cfg = infer_cfg['retriever'].copy()
            retriever_cfg['dataset'] = dataset
            retriever = ICL_RETRIEVERS.build(retriever_cfg)

            inferencer = self._build_inferencer(model, model_cfg, dataset_cfg,
                                                max_workers)
            if hasattr(inferencer, 'progress_tracker'):
                inferencer.progress_tracker = progress

            out_path = get_infer_output_path(
                model_cfg, dataset_cfg, osp.join(self.work_dir, 'predictions'))
            out_dir, out_file = osp.split(out_path)
            mkdir_or_exist(out_dir)

            infer_kwargs = dict(
                output_json_filepath=out_dir,
                output_json_filename=out_file,
            )
            if 'prompt_template' in infer_cfg:
                infer_kwargs['prompt_template'] = ICL_PROMPT_TEMPLATES.build(
                    infer_cfg['prompt_template'])
            if 'ice_template' in infer_cfg:
                infer_kwargs['ice_template'] = ICL_PROMPT_TEMPLATES.build(
                    infer_cfg['ice_template'])
            inferencer.inference(retriever, **infer_kwargs)
        except Exception as e:
            import traceback
            status.update(status='fail')
            trace = "\n".join(traceback.format_exception(e))
            self.logger.error(f'Infer failed with\n{trace}')
        else:
            status.update(status='done')

    def _remaining_total(self, running: List[_RunningTask],
                         max_pending_samples: int) -> int:
        total = 0
        for item in running:
            remaining = item.progress.remaining()
            if remaining is None:
                return max_pending_samples
            total += remaining
        return total

    def _run_task_group(self, model_cfg, tasks, tokens, max_workers):
        pending = list(tasks)
        running: List[_RunningTask] = []
        last_log = 0.0
        abbr_counts = {}
        for dataset_cfg in pending:
            abbr = dataset_cfg.get('abbr', 'task')
            abbr_counts[abbr] = abbr_counts.get(abbr, 0) + 1

        status_entries: Dict[str, InferStatusManager] = {}
        for dataset_cfg in pending:
            task_name = task_abbr_from_cfg({
                'models': [model_cfg],
                'datasets': [[dataset_cfg]],
            })
            abbr = dataset_cfg.get('abbr', 'task')
            status_entries[task_name] = InferStatusManager(
                self.work_dir, model_cfg, dataset_cfg)
            status_entries[task_name].update(status='pending')

        max_parallel_ds = min(len(pending), 32) or 1
        with ThreadPoolExecutor(max_workers=max_parallel_ds) as executor:
            while pending or running:
                # collect finished tasks
                finished = []
                for item in running:
                    if item.future.done():
                        finished.append(item)
                        self.logger.info(f'Finished dataset task {item.name}')
                if finished:
                    running = [item for item in running if item not in finished]

                # Start next sub task
                # Load next dataset with 5% overhead.
                max_pending_samples = max(1, int(max_workers * 1.05))
                remaining_total = self._remaining_total(running, max_pending_samples)
                while pending and remaining_total < max_pending_samples:
                    dataset_cfg = pending.pop(0)
                    task_name = task_abbr_from_cfg({
                        'models': [model_cfg],
                        'datasets': [[dataset_cfg]],
                    })
                    progress = _ProgressTracker(task_name)
                    future = executor.submit(
                        self._run_dataset_task,
                        model_cfg,
                        dataset_cfg,
                        tokens,
                        max_workers,
                        progress,
                        status_entries[task_name],
                    )
                    running.append(_RunningTask(task_name, progress, future))
                    entry = status_entries.get(task_name)
                    if entry is not None:
                        entry.update(status='running')
                    self.logger.info(f'Start dataset task {task_name}')
                    remaining_total = self._remaining_total(
                        running, max_pending_samples)

                now = time.time()
                if now - last_log >= self.log_interval and running:
                    status = ', '.join(
                        f'{item.name}:{item.progress.completed}/'
                        f'{item.progress.total if item.progress.total is not None else "?"}'
                        for item in running)
                    self.logger.info(f'Running tasks: {status}')
                    last_log = now

                for item in finished + running:
                    entry = status_entries.get(item.name)
                    if entry is not None:
                        entry.update(
                            total=item.progress.total,
                            completed=item.progress.completed,
                        )

                if running:
                    time.sleep(self.poll_interval)


def parse_args():
    parser = argparse.ArgumentParser(description='Model Inferencer')
    parser.add_argument('config', help='Config file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    start_time = time.time()
    inferencer = OpenICLInferTaskConcurrent(cfg)
    inferencer.run()
    end_time = time.time()
    get_logger().info(f'time elapsed: {end_time - start_time:.2f}s')
