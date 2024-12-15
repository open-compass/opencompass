import argparse
import os
import os.path as osp
import random
import sys
import time
from typing import Any
from tqdm.asyncio import tqdm

from mmengine.config import Config, ConfigDict
import inspect
from mmengine.utils import mkdir_or_exist

from opencompass.registry import (ICL_INFERENCERS, ICL_PROMPT_TEMPLATES,
                                  ICL_RETRIEVERS, TASKS)
from opencompass.tasks.base import BaseTask
from opencompass.utils import (build_dataset_from_cfg, build_model_from_cfg,
                               get_infer_output_path, get_logger,
                               task_abbr_from_cfg)
from opencompass.openicl.icl_inferencer.icl_gen_async_inferencer import AsyncGenInferencer
from opencompass.openicl.icl_inferencer.icl_chat_async_inferencer import AsyncChatInferencer
from opencompass.openicl.icl_inferencer import GenInferencer, ChatInferencer
from concurrent.futures import ThreadPoolExecutor
import asyncio
import resource
from more_itertools import consume


soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, hard))


@TASKS.register_module()
class OpenICLAsyncInferTask(BaseTask):
    """OpenICL Inference Task.

    This task is used to run the inference process.
    """

    name_prefix = 'OpenICLInfer'
    log_subdir = 'logs/infer'
    output_subdir = 'predictions'

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        run_cfg = self.model_cfgs[0].get('run_cfg', {})
        self.nproc = run_cfg.get('nproc_per_worker', 16)

    def get_command(self, cfg_path, template) -> str:
        # TODO:
        raise NotImplementedError()
        return ""

    async def run(self):  # type: ignore
        _dataset_cfgs = []
        infer_cfgs = []
        sub_cfgs = []
        datasets = []
        model_cfgs = []
        for model_cfg, dataset_cfgs in zip(self.model_cfgs, self.dataset_cfgs):
            self.max_out_len = model_cfg.get('max_out_len', None)
            self.batch_size = model_cfg.get('batch_size', None)
            self.min_out_len = model_cfg.get('min_out_len', None)

            for dataset_cfg in dataset_cfgs:
                self.dataset_cfg = dataset_cfg
                out_path = get_infer_output_path(
                    model_cfg, dataset_cfg,
                    osp.join(self.work_dir, 'predictions'))

                if osp.exists(out_path):
                    continue
                _dataset_cfgs.append(dataset_cfg)
                datasets.append(build_dataset_from_cfg(dataset_cfg))
                infer_cfgs.append(dataset_cfg['infer_cfg'])
                model_cfgs.append(model_cfg)
                sub_cfg = {
                    'models': [model_cfg],
                    'datasets': [[dataset_cfg]],
                }
                sub_cfgs.append(sub_cfg)

        tasks = []
        args = list(zip(_dataset_cfgs, infer_cfgs, datasets, model_cfgs, sub_cfgs))
        for arg in tqdm(
            args,
            total=len(args),
            desc=f"Starting building tasks..."
        ):
            tasks.append(asyncio.create_task(self._inference(*arg)))

        bar = tqdm(desc="Inferencing...", total=len(tasks))
        bar.refresh()

        while tasks:
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for _ in done:
                bar.update()
                bar.refresh()

        # TODO: Needs a debug mode
        # for arg in zip(_dataset_cfgs, infer_cfgs, datasets, model_cfgs, sub_cfgs):
        #     await self._inference(*arg)

    async def _inference(self, dataset_cfg, infer_cfg, dataset, model_cfg, sub_cfg):
        model = build_model_from_cfg(model_cfg)
        assert hasattr(infer_cfg, 'ice_template') or hasattr(infer_cfg, 'prompt_template'), \
            'Both ice_template and prompt_template cannot be None simultaneously.'  # noqa: E501

        infer_kwargs: dict = {}
        if hasattr(infer_cfg, 'ice_template'):
            ice_template = ICL_PROMPT_TEMPLATES.build(
                infer_cfg['ice_template'])
            infer_kwargs['ice_template'] = ice_template

        if hasattr(infer_cfg, 'prompt_template'):
            prompt_template = ICL_PROMPT_TEMPLATES.build(
                infer_cfg['prompt_template'])
            infer_kwargs['prompt_template'] = prompt_template

        retriever_cfg = infer_cfg['retriever'].copy()
        retriever_cfg['dataset'] = dataset
        retriever = ICL_RETRIEVERS.build(retriever_cfg)

        # set inferencer's default value according to model's config'
        inferencer_cfg: dict = infer_cfg['inferencer']
        inferencer_cfg['model'] = model
        inferencer_cfg['max_seq_len'] = model_cfg.get('max_seq_len')

        infer_type = inferencer_cfg["type"]
        if inspect.isclass(infer_type):
            infer_name = infer_type.__name__
        else:
            infer_name = infer_type

        if infer_name.split(".")[-1] == "ChatInferencer":
            inferencer_cfg["type"] = AsyncChatInferencer

        elif infer_name.split(".")[-1] == "GenInferencer":
            inferencer_cfg["type"] = AsyncGenInferencer

        inferencer_cfg.setdefault('max_out_len', self.max_out_len)
        inferencer_cfg.setdefault('min_out_len', self.min_out_len)
        inferencer_cfg.setdefault('batch_size', self.batch_size)
        inferencer = ICL_INFERENCERS.build(inferencer_cfg)

        out_path = get_infer_output_path(
            model_cfg, dataset_cfg,
            osp.join(self.work_dir, 'predictions'))
        out_dir, out_file = osp.split(out_path)
        mkdir_or_exist(out_dir)

        infer_kwargs['output_json_filepath'] = out_dir
        infer_kwargs['output_json_filename'] = out_file

        await inferencer.inference(retriever, **infer_kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description='Model Inferencer')
    parser.add_argument('config', help='Config file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # TODO:
    raise NotImplementedError()
