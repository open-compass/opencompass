"""Parallel Chat Inferencer."""
import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import mmengine

from opencompass.registry import ICL_INFERENCERS

from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever
from ..utils.logging import get_logger
from .icl_chat_inferencer import ChatInferencer

logger = get_logger(__name__)


@ICL_INFERENCERS.register_module()
class ParallelChatInferencer(ChatInferencer):
    """Parallel chat inferencer with thread pool over samples."""

    def __init__(
            self,
            model,
            output_json_filepath: Optional[str] = './icl_inference_output',
            output_json_filename: Optional[str] = 'predictions',
            save_every: Optional[int] = 1,
            infer_mode: str = 'last',
            max_out_len: int = 512,
            max_infer_workers: Optional[int] = None,
            **kwargs) -> None:
        super().__init__(
            model=model,
            output_json_filename=output_json_filename,
            output_json_filepath=output_json_filepath,
            save_every=save_every,
            infer_mode=infer_mode,
            max_out_len=max_out_len,
            **kwargs,
        )
        self.max_infer_workers = max_infer_workers
        self.progress_tracker = None

    def _resolve_max_workers(self) -> int:
        if self.max_infer_workers is not None:
            return self.max_infer_workers
        max_workers = getattr(self.model, 'max_workers', None)
        if max_workers is not None:
            return max_workers
        cpu_count = os.cpu_count() or 1
        return min(32, cpu_count + 4)

    def _progress_update(self, count: int = 1) -> None:
        if self.progress_tracker is not None:
            self.progress_tracker.incr(count)

    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None) -> dict:
        output_handler = self.HandlerType()

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        ice_idx_list = retriever.retrieve()

        chat_list = self.get_chat_list(
            ice_idx_list,
            retriever,
            prompt_template=prompt_template,
        )

        total_samples = len(chat_list)
        if self.progress_tracker is not None:
            self.progress_tracker.set_total(total_samples)

        todo = list(range(total_samples))
        tmp_json_filepath = os.path.join(output_json_filepath,
                                         'tmp_' + output_json_filename)
        if osp.exists(tmp_json_filepath):
            try:
                tmp_result_dict = mmengine.load(tmp_json_filepath)
            except Exception:
                pass
            else:
                output_handler.results_dict = tmp_result_dict
                todo = [i for i in todo if str(i) not in tmp_result_dict.keys()]
        if self.progress_tracker is not None:
            self.progress_tracker.set_completed(total_samples - len(todo))

        chats = [chat_list[i] for i in todo]

        logger.info('Starting parallel chat inference process...')

        def _infer_one(chat, idx):
            local_handler = self.HandlerType()
            if self.infer_mode == 'last':
                self.infer_last(chat, idx, local_handler)
            elif self.infer_mode == 'every':
                self.infer_every(chat, idx, local_handler)
            elif self.infer_mode == 'every_with_gt':
                self.infer_every_with_gt(chat, idx, local_handler)
            return local_handler.results_dict

        max_workers = self._resolve_max_workers()
        completed = total_samples - len(todo)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_infer_one, chat, idx)
                for idx, chat in zip(todo, chats)
            ]
            for future in as_completed(futures):
                result_dict = future.result()
                output_handler.results_dict.update(result_dict)
                delta = len(result_dict)
                completed += delta
                self._progress_update(delta)
                if (self.save_every is not None
                        and completed % self.save_every == 0
                        and self.is_main_process):
                    output_handler.write_to_json(output_json_filepath,
                                                 'tmp_' + output_json_filename)

        if self.is_main_process:
            os.makedirs(output_json_filepath, exist_ok=True)
            output_handler.write_to_json(output_json_filepath,
                                         output_json_filename)
            if osp.exists(tmp_json_filepath):
                os.remove(tmp_json_filepath)

        return output_handler.results_dict
