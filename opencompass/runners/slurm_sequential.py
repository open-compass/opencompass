import os
import os.path as osp
import re
import subprocess
import time
import traceback
from functools import partial
from multiprocessing import Pipe, Pool
from typing import Any, Dict, List, Tuple

import mmengine
from mmengine.config import ConfigDict
from tqdm import tqdm

from opencompass.registry import RUNNERS, TASKS
from opencompass.utils import get_logger

from .base import BaseRunner


@RUNNERS.register_module()
class SlurmSequentialRunner(BaseRunner):
    """Distributed runner based on Slurm. It will launch tasks in parallel
    using `srun` command.

    This runner launches tasks one by one for execution. A new task will only
    be launched when and only when max_num_workers is not met, and the previous
    task has been successfully allocated to a machine. Therefore, unlike the
    `SlurmRunner`, at most only one task will be in the PENDING status at the
    same time during a run, making the random_sleep strategy no longer
    necessary. In addition, this runner also includes a feature to
    automatically kill all jobs by the job_id on exit.

    The runner will obtain the job_id by reading the srun output similar to
    `srun: Job 123456 scheduled successfully!`. If the output of srun does not
    match this pattern, the runner will not work properly.

    Args:
        task (ConfigDict): Task type config.
        max_num_workers (int): Max number of workers to run in parallel.
            Defaults to 32.
        retry (int): Number of retries if the job failed. Defaults to 2.
        partition (str): Slurm partition name. Defaults to None.
        quotatype (str): Slurm quota type. Defaults to None.
        qos (str): Slurm quality of service. Defaults to None.
        debug (bool): Whether to run in debug mode. Defaults to False.
        lark_bot_url (str): Lark bot url. Defaults to None.
    """

    def __init__(self,
                 task: ConfigDict,
                 task_prefix: str = '',
                 max_num_workers: int = 32,
                 retry: int = 2,
                 partition: str = None,
                 quotatype: str = None,
                 qos: str = None,
                 debug: bool = False,
                 lark_bot_url: str = None):
        super().__init__(task=task, debug=debug, lark_bot_url=lark_bot_url)
        self.max_num_workers = max_num_workers
        self.retry = retry
        self.partition = partition
        self.quotatype = quotatype
        self.qos = qos
        self.task_prefix = task_prefix

        logger = get_logger()
        if self.quotatype in ['spot', 'auto']:
            logger.warning(
                'Quotatype spot or auto may cause stability issues, '
                'reserved is recommended.')

    def launch(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        if not self.debug:
            return self._launch_wo_debug(tasks)
        else:
            return [self._launch(task) for task in tasks]

    def _launch_wo_debug(self,
                         tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        launched_bar = tqdm(total=len(tasks), desc='Launched')
        finished_bar = tqdm(total=len(tasks), desc='Finished')
        job_ids = []
        status = []

        def _update(result):
            finished_bar.update()
            status.append(result)
            return result

        def _err_update(err):
            finished_bar.update()
            traceback.print_exc()
            status.append(('', -1))

        try:
            parent_conns = []
            num_workers = max(min(self.max_num_workers, len(tasks)), 1)
            with Pool(processes=num_workers) as pool:
                for task in tasks:
                    parent_conn, child_conn = Pipe()
                    _ = pool.apply_async(self._launch,
                                         kwds={
                                             'cfg': task,
                                             'child_conn': child_conn
                                         },
                                         callback=_update,
                                         error_callback=_err_update)
                    time.sleep(0.5)

                    job_id = parent_conn.recv()
                    launched_bar.update()
                    parent_conns.append(parent_conn)
                    job_ids.append(job_id)

                pool.close()
                pool.join()
            return status
        except KeyboardInterrupt:
            raise
        finally:
            launched_bar.close()
            finished_bar.close()
            for parent_conn in parent_conns:
                while parent_conn.poll():
                    try:
                        job_id = parent_conn.recv()
                        job_ids.append(job_id)
                    except EOFError:
                        break
                parent_conn.close()

            for job_id in tqdm(job_ids, desc='clear sruns'):
                if job_id is None:
                    continue
                cmd = f'scancel {job_id}'
                p = subprocess.Popen(cmd,
                                     shell=True,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
                p.wait()

    def _launch(self, cfg: ConfigDict, child_conn: Pipe = None):
        logger = get_logger()

        task = TASKS.build(dict(cfg=cfg, type=self.task_cfg['type']))
        num_gpus = task.num_gpus
        task_name = task.name
        task_name = self.task_prefix + task_name

        # Dump task config to file
        mmengine.mkdir_or_exist('tmp/')
        param_file = f'tmp/{os.getpid()}_params.py'
        process = None
        try:
            cfg.dump(param_file)

            # Build up slurm command
            tmpl = 'srun'
            if self.partition:
                tmpl += f' -p {self.partition}'
            if self.quotatype:
                tmpl += f' --quotatype={self.quotatype}'
            if self.qos:
                tmpl += f' --qos={self.qos}'
            if num_gpus > 0:
                tmpl += f' --gres=gpu:{num_gpus}'
            tmpl += f" -N1 -J '{task_name[:512]}'" + ' {task_cmd}'
            get_cmd = partial(task.get_command,
                              cfg_path=param_file,
                              template=tmpl)
            cmd = get_cmd()

            logger.debug(f'Running command: {cmd}')

            retry = self.retry
            output_paths = task.get_output_paths()

            if self.debug:
                while True:
                    process = subprocess.Popen(cmd, shell=True, text=True)
                    process.communicate()
                    process.wait()
                    if self._job_failed(process.returncode, output_paths):
                        if retry > 0:
                            logger.warning(
                                f'task {task_name} failed, retrying...')
                            retry -= 1
                            cmd = get_cmd()
                        else:
                            break
                    else:
                        break
            else:
                out_path = task.get_log_path(file_extension='out')
                mmengine.mkdir_or_exist(osp.split(out_path)[0])
                stdout = open(out_path, 'w', encoding='utf-8')
                stderr = subprocess.PIPE
                while True:
                    process = subprocess.Popen(cmd,
                                               shell=True,
                                               text=True,
                                               stdout=stdout,
                                               stderr=stderr)
                    job_id = None
                    while True:
                        line = process.stderr.readline()
                        if not line:
                            break
                        match = re.search(
                            r'srun: Job (\d+) scheduled successfully!', line)
                        if match and job_id is None:
                            job_id = match.group(1)
                            child_conn.send(job_id)
                        stdout.write(line)
                    process.wait()
                    if self._job_failed(process.returncode, output_paths):
                        if retry > 0:
                            retry -= 1
                            cmd = get_cmd()
                        else:
                            logger.warning(
                                f'task {task_name} fail, see\n{out_path}')
                            break
                    else:
                        break
        except KeyboardInterrupt:
            raise
        finally:
            # Clean up
            if child_conn is not None:
                child_conn.send(None)
                child_conn.close()
            if process is not None:
                process.kill()
            os.remove(param_file)
        return task_name, process.returncode

    def _job_failed(self, return_code: int, output_paths: List[str]) -> bool:
        return return_code != 0 or not all(
            osp.exists(output_path) for output_path in output_paths)
