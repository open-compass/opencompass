import copy
import os
from abc import abstractmethod
from typing import List

from mmengine.config import ConfigDict

from opencompass.utils import get_infer_output_path, task_abbr_from_cfg


class BaseTask:
    """Base class for all tasks. There are two ways to run the task:
    1. Directly by calling the `run` method.
    2. Calling the `get_command` method to get the command,
        and then run the command in the shell.

    Args:
        cfg (ConfigDict): Config dict.
    """

    # The prefix of the task name.
    name_prefix: str = None
    # The subdirectory of the work directory to store the log files.
    log_subdir: str = None
    # The subdirectory of the work directory to store the output files.
    output_subdir: str = None

    def __init__(self, cfg: ConfigDict):
        cfg = copy.deepcopy(cfg)
        self.cfg = cfg
        self.model_cfgs = cfg['models']
        self.dataset_cfgs = cfg['datasets']
        self.work_dir = cfg['work_dir']

    @abstractmethod
    def run(self):
        """Run the task."""

    @abstractmethod
    def get_command(self, cfg_path, template) -> str:
        """Get the command template for the task.

        Args:
            cfg_path (str): The path to the config file of the task.
            template (str): The template which have '{task_cmd}' to format
                the command.
        """

    @property
    def name(self) -> str:
        return self.name_prefix + task_abbr_from_cfg(
            {
                'models': self.model_cfgs,
                'datasets': self.dataset_cfgs
            })

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.cfg})'

    def get_log_path(self, file_extension: str = 'json') -> str:
        """Get the path to the log file.

        Args:
            file_extension (str): The file extension of the log file.
                Default: 'json'.
        """
        return get_infer_output_path(
            self.model_cfgs[0], self.dataset_cfgs[0][0],
            os.path.join(self.work_dir, self.log_subdir), file_extension)

    def get_output_paths(self, file_extension: str = 'json') -> List[str]:
        """Get the paths to the output files. Every file should exist if the
        task succeeds.

        Args:
            file_extension (str): The file extension of the output files.
                Default: 'json'.
        """
        output_paths = []
        for model, datasets in zip(self.model_cfgs, self.dataset_cfgs):
            for dataset in datasets:
                output_paths.append(
                    get_infer_output_path(
                        model, dataset,
                        os.path.join(self.work_dir, self.output_subdir),
                        file_extension))
        return output_paths
