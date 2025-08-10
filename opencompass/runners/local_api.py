import logging
import os
import os.path as osp
import subprocess
import sys
import time
import traceback
from multiprocessing import Manager, Pool
from multiprocessing.managers import SyncManager
from typing import Any, Dict, List, Tuple

import mmengine
from mmengine.config import ConfigDict
from tqdm import tqdm

from opencompass.registry import RUNNERS, TASKS
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.base import BaseTask
from opencompass.utils import (build_dataset_from_cfg, build_model_from_cfg,
                               get_infer_output_path, get_logger,
                               task_abbr_from_cfg)

from .base import BaseRunner


def monkey_run(self, tokens: SyncManager.Semaphore):
    """Hack for infer task run, add tokens for multiprocess."""
    self.logger.info(f'Task {task_abbr_from_cfg(self.cfg)}')
    for model_cfg, dataset_cfgs in zip(self.model_cfgs, self.dataset_cfgs):
        self.max_out_len = model_cfg.get('max_out_len', None)
        self.min_out_len = model_cfg.get('min_out_len', None)
        self.batch_size = model_cfg.get('batch_size', None)
        self.model = build_model_from_cfg(model_cfg)
        # add global tokens for concurrents
        assert self.model.is_api, 'Only API model is supported.'
        self.model.tokens = tokens

        for dataset_cfg in dataset_cfgs:
            self.model_cfg = model_cfg
            self.dataset_cfg = dataset_cfg
            self.infer_cfg = self.dataset_cfg['infer_cfg']
            self.dataset = build_dataset_from_cfg(self.dataset_cfg)
            self.sub_cfg = {
                'models': [self.model_cfg],
                'datasets': [[self.dataset_cfg]],
            }
            out_path = get_infer_output_path(
                self.model_cfg, self.dataset_cfg,
                osp.join(self.work_dir, 'predictions'))
            if osp.exists(out_path):
                continue
            self._inference()


old_stdout = sys.stdout
old_stderr = sys.stderr


def redirect_std_to_file(filename: str):
    """Redirect stdout and stderr, also change logger stream handler."""
    f = open(filename, 'w', encoding='utf-8')
    sys.stdout = f
    sys.stderr = f
    # change logger stream handler as well
    logger = get_logger()
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler):
            h.stream = sys.stdout
    # special treat for icl_gen_inferencer logger
    gen_logger = logging.getLogger(
        'opencompass.openicl.icl_inferencer.icl_gen_inferencer')
    for h in gen_logger.handlers:
        if isinstance(h, logging.StreamHandler):
            h.stream = sys.stdout


def reset_std():
    """Reset stdout and stderr, also change logger stream handler."""
    sys.stdout.close()
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    # change logger stream handler as well
    logger = get_logger()
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler):
            h.stream = sys.stdout
    # special treat for icl_gen_inferencer logger
    gen_logger = logging.getLogger(
        'opencompass.openicl.icl_inferencer.icl_gen_inferencer')
    for h in gen_logger.handlers:
        if isinstance(h, logging.StreamHandler):
            h.stream = sys.stdout


def launch(task: BaseTask, tokens: SyncManager.Semaphore):
    """Launch a single task.

    Args:
        task (BaseTask): Task to launch.
        tokens (SyncManager.Semaphore): Multiprocessing semaphore
            for every subprocess to follow.

    Returns:
        tuple[str, int]: Task name and exit code.
    """

    task_name = task.name
    returncode = 0
    logger = get_logger()

    try:
        # get log file and redirect stdout and stderr
        out_path = task.get_log_path(file_extension='out')
        mmengine.mkdir_or_exist(osp.split(out_path)[0])
        redirect_std_to_file(out_path)

        # start infer with monkey_run
        start_time = time.time()
        inferencer = OpenICLInferTask(task.cfg)
        origin_run = inferencer.run
        inferencer.run = monkey_run
        inferencer.run(inferencer, tokens)
        inferencer.run = origin_run
        end_time = time.time()
        logger.info(f'time elapsed: {end_time - start_time:.2f}s')
    except Exception:
        # print trace back in target file
        traceback.print_exc()
        # reset stdout and stderr
        reset_std()
        logger.error(f'task {task_name} fail, see\n{out_path}')
        returncode = 1
    else:
        # reset stdout and stderr
        reset_std()
    return task_name, returncode


def submit(task, type, tokens):
    """Helper for launch the task."""
    task = TASKS.build(dict(cfg=task, type=type))
    tqdm.write(f'Launch {task.name} on CPU ')

    res = launch(task, tokens)
    return res


@RUNNERS.register_module()
class LocalAPIRunner(BaseRunner):
    """Local API Runner. Start tasks by local python.

    The query per second cannot guarantee the number of concurrents, therefore
    Supported concurrent users with multiple tasks. Applied for those apis
    which has a restriction on concurrent numbers.

    Args:
        task (ConfigDict): Task type config.
        concurrent_users (int): Max number of concurrent workers to request
            the resources.
        max_num_workers (int): Max number of workers to run in parallel.
            Defaults to 16.
        debug (bool): Whether to run in debug mode.
        lark_bot_url (str): Lark bot url.
    """

    def __init__(self,
                 task: ConfigDict,
                 concurrent_users: int,
                 max_num_workers: int = 16,
                 debug: bool = False,
                 lark_bot_url: str = None):
        super().__init__(task=task, debug=debug, lark_bot_url=lark_bot_url)
        self.max_num_workers = max_num_workers
        self.concurrent_users = concurrent_users
        assert task['type'] in [
            'OpenICLInferTask',
            'opencompass.tasks.OpenICLInferTask',
        ], 'Only supported for api infer task.'

    def launch(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Launch multiple tasks.

        Args:
            tasks (list[dict]): A list of task configs, usually generated by
                Partitioner.

        Returns:
            list[tuple[str, int]]: A list of (task name, exit code).
        """
        status = []
        if self.debug:
            # fall back to LocalRunner debug mode
            for task in tasks:
                task = TASKS.build(dict(cfg=task, type=self.task_cfg['type']))
                task_name = task.name
                # get cmd
                mmengine.mkdir_or_exist('tmp/')
                param_file = f'tmp/{os.getpid()}_params.py'
                try:
                    task.cfg.dump(param_file)
                    cmd = task.get_command(cfg_path=param_file,
                                           template='{task_cmd}')
                    # run in subprocess if starts with torchrun etc.
                    if cmd.startswith('python'):
                        task.run()
                    else:
                        subprocess.run(cmd, shell=True, text=True)
                finally:
                    os.remove(param_file)
                status.append((task_name, 0))
        else:

            pbar = tqdm(total=len(tasks))

            get_logger().info('All the logs and processes for each task'
                              ' should be checked in each infer/.out file.')
            with Manager() as manager:
                tokens = manager.Semaphore(self.concurrent_users)
                # pbar update has visualization issue when direct
                # update pbar in callback, need an extra counter
                pbar_counter = manager.Value('i', 0)
                status = []

                def update(args):
                    """Update pbar counter when callback."""
                    pbar_counter.value += 1
                    status.append(args)

                with Pool(processes=self.max_num_workers) as pool:
                    for task in tasks:
                        pool.apply_async(submit,
                                         (task, self.task_cfg['type'], tokens),
                                         callback=update)
                    pool.close()

                    # update progress bar
                    while True:
                        cur_count = pbar_counter.value
                        if cur_count > pbar.n:
                            pbar.update(cur_count - pbar.n)
                        # break when all the task finished
                        if cur_count >= pbar.total:
                            pbar.close()
                            break
                        # sleep to lower the usage
                        time.sleep(1)

                    pool.join()
        return status
