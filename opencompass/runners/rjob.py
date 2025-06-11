import os
import os.path as osp
import random
import subprocess
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import mmengine
from mmengine.config import ConfigDict
from mmengine.utils import track_parallel_progress

from opencompass.registry import RUNNERS, TASKS
from opencompass.utils import get_logger

from .base import BaseRunner


@RUNNERS.register_module()
class RJOBRunner(BaseRunner):
    """Runner for submitting jobs via rjob bash script. Structure similar to DLC/VOLC runners.

    Args:
        task (ConfigDict): Task type config.
        rjob_cfg (ConfigDict): rjob相关配置。
        max_num_workers (int): 最大并发数。
        retry (int): 失败重试次数。
        debug (bool): 是否debug模式。
        lark_bot_url (str): Lark通知。
        keep_tmp_file (bool): 是否保留临时文件。
    """

    def __init__(
        self,
        task: ConfigDict,
        rjob_cfg: ConfigDict,
        max_num_workers: int = 32,
        retry: int = 2,
        debug: bool = False,
        lark_bot_url: str = None,
        keep_tmp_file: bool = True,
    ):
        super().__init__(task=task, debug=debug, lark_bot_url=lark_bot_url)
        self.rjob_cfg = rjob_cfg
        self.max_num_workers = max_num_workers
        self.retry = retry
        self.keep_tmp_file = keep_tmp_file

    # def normalize_task_name(self, name, max_length=60):
    #     import re
    #     # 只保留字母、数字、-，其他都替换成 -
    #     name = re.sub(r'[^a-zA-Z0-9-]', '-', name)
    #     # 去掉连续的 -
    #     name = re.sub(r'-+', '-', name)
    #     # 去掉开头和结尾的 -
    #     name = name.strip('-')
    #     # 控制长度
    #     if len(name) > max_length:
    #         name = name[:max_length]
    #     # 保证长度至少为 1
    #     if not name:
    #         name = 'task'
    #     return name.lower()

    def launch(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Launch multiple tasks."""
        if not self.debug:
            status = track_parallel_progress(
                self._launch,
                tasks,
                nproc=self.max_num_workers,
                keep_order=False,
            )
        else:
            status = [self._launch(task, random_sleep=False) for task in tasks]
        return status

    def _run_task(self, task_name, log_path, poll_interval=60):
        """轮询 rjob 状态，直到 active 和 pending 都为 0 时 break。如果没有 dict 行，直接 break。"""
        import ast
        logger = get_logger()
        status = None
        time.sleep(10)
        while True:
            get_cmd = f"rjob get {task_name}"
            get_result = subprocess.run(get_cmd, shell=True, text=True, capture_output=True)
            output = get_result.stdout
            if log_path:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n[rjob get] {output}\n")

            found_dict = False
            for line in output.splitlines():
                if '{' in line and '}' in line:
                    try:
                        d = ast.literal_eval(line[line.index('{'):line.index('}')+1])
                        found_dict = True
                        if d.get('active', 0) != 0 or d.get('pending', 0) != 0:
                            break
                        else:
                            status = 'FINISHED'
                            logger.info(f'[RJOB] Final status returned: {status}')
                            return status
                    except Exception as e:
                        pass
            if found_dict:
                time.sleep(poll_interval)
                continue
            break
        logger.info(f'[RJOB] Final status returned: {status}')
        return status

    def _launch(self, cfg: ConfigDict, random_sleep: Optional[bool] = None):
        """Launch a single task via rjob bash script."""
        if random_sleep is None:
            random_sleep = self.max_num_workers > 32
        task = TASKS.build(dict(cfg=cfg, type=self.task_cfg['type']))
        num_gpus = task.num_gpus
        # 合法化 task_name
        # task_name = model_abbr_from_cfg(self.)
        # task_name = self.normalize_task_name(task.name)
        import uuid
        task_name = "opencompass-" + str(uuid.uuid4())
        logger = get_logger()
        logger.info(f'Task name: {task_name}')

        # 生成临时参数文件
        pwd = os.getcwd()
        mmengine.mkdir_or_exist('tmp/')

        uuid_str = str(uuid.uuid4())
        param_file = f'{pwd}/tmp/{uuid_str}_params.py'
        try:
            cfg.dump(param_file)

            # 拼接 rjob submit 命令参数
            args = []
            # 基本参数
            args.append(f'--name={task_name}')
            if num_gpus > 0:
                args.append(f'--gpu={num_gpus}')
            if hasattr(task, 'memory'):
                args.append(f'--memory={getattr(task, "memory")}')
            elif self.rjob_cfg.get('memory', 300000):
                args.append(f'--memory={self.rjob_cfg["memory"]}')
            if hasattr(task, 'cpu'):
                args.append(f'--cpu={getattr(task, "cpu")}')
            elif self.rjob_cfg.get('cpu', 16):
                args.append(f'--cpu={self.rjob_cfg["cpu"]}')
            if self.rjob_cfg.get('charged_group'):
                args.append(f'--charged-group={self.rjob_cfg["charged_group"]}')
            if self.rjob_cfg.get('private_machine'):
                args.append(f'--private-machine={self.rjob_cfg["private_machine"]}')
            if self.rjob_cfg.get('mount'):
                # 支持多个 mount
                mounts = self.rjob_cfg['mount']
                if isinstance(mounts, str):
                    mounts = [mounts]
                for m in mounts:
                    args.append(f'--mount={m}')
            if self.rjob_cfg.get('image'):
                args.append(f'--image={self.rjob_cfg["image"]}')
            if self.rjob_cfg.get('replicas'):
                args.append(f'-P {self.rjob_cfg["replicas"]}')
            if self.rjob_cfg.get('host_network'):
                args.append(f'--host-network={self.rjob_cfg["host_network"]}')
            # 环境变量
            envs = self.rjob_cfg.get('env', {})
            if isinstance(envs, dict):
                for k, v in envs.items():
                    args.append(f'-e {k}={v}')
            elif isinstance(envs, list):
                for e in envs:
                    args.append(f'-e {e}')
            # 额外参数
            if self.rjob_cfg.get('extra_args'):
                args.extend(self.rjob_cfg['extra_args'])

            # 启动命令通过 task.get_command 得到，兼容 template
            tmpl = '{task_cmd}'
            get_cmd = partial(task.get_command, cfg_path=param_file, template=tmpl)
            entry_cmd = get_cmd()
            entry_cmd = f'bash -c "cd {pwd} && {entry_cmd}"'
            # 拼接完整命令
            cmd = f"rjob submit {' '.join(args)} -- {entry_cmd}"

            logger = get_logger()
            logger.info(f'Running command: {cmd}')

            # 日志输出
            if self.debug:
                out_path = None
            else:
                out_path = task.get_log_path(file_extension='out')
                mmengine.mkdir_or_exist(osp.split(out_path)[0])

            if random_sleep:
                time.sleep(random.randint(0, 10))

            retry = self.retry
            while retry > 0:
                # 只 submit，不轮询
                result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
                logger.info(f'Command output: {result.stdout}')
                if result.stderr:
                    logger.error(f'Command error: {result.stderr}')
                logger.info(f'Return code: {result.returncode}')
                if result.returncode == 0:
                    break
                retry -= 1
                time.sleep(2)
            if result.returncode != 0:
                # submit失败，直接返回
                return task_name, result.returncode

            # submit成功，进入轮询
            status = self._run_task(task_name, out_path)
            output_paths = task.get_output_paths()
            returncode = 0 if status == 'FINISHED' else 1
            if self._job_failed(returncode, output_paths):
                returncode = 1
        finally:
            if not self.keep_tmp_file:
                os.remove(param_file)

        return task_name, returncode

    def _job_failed(self, return_code: int, output_paths: List[str]) -> bool:
        return return_code != 0 or not all(
            osp.exists(output_path) for output_path in output_paths)
