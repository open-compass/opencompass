"""Parallel Generation Inferencer."""
import copy
import inspect
import json
import os
import os.path as osp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import mmengine

from opencompass.models.openai_api import OpenAISDK
from opencompass.registry import ICL_INFERENCERS
from opencompass.utils import batched

from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever
from ..utils.logging import get_logger
from .icl_base_inferencer import GenInferencerOutputHandler
from .icl_gen_inferencer import GenInferencer

logger = get_logger(__name__)


@ICL_INFERENCERS.register_module()
class ParallelGenInferencer(GenInferencer):
    """Parallel generation inferencer with thread pool over samples."""

    def __init__(
            self,
            model,
            max_out_len: int,
            stopping_criteria: List[str] = [],
            max_seq_len: Optional[int] = None,
            min_out_len: Optional[int] = None,
            gen_field_replace_token: Optional[str] = '',
            output_json_filepath: Optional[str] = './icl_inference_output',
            output_json_filename: Optional[str] = 'predictions',
            save_every: Optional[int] = 1,
            max_infer_workers: Optional[int] = None,
            **kwargs) -> None:
        super().__init__(
            model=model,
            max_out_len=max_out_len,
            stopping_criteria=stopping_criteria,
            max_seq_len=max_seq_len,
            min_out_len=min_out_len,
            batch_size=1,
            gen_field_replace_token=gen_field_replace_token,
            output_json_filepath=output_json_filepath,
            output_json_filename=output_json_filename,
            save_every=save_every,
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
                  output_json_filename: Optional[str] = None) -> List:
        output_handler = GenInferencerOutputHandler()

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        ice_idx_list = retriever.retrieve()

        prompt_list = self.get_generation_prompt_list_from_retriever_indices(
            ice_idx_list,
            retriever,
            self.gen_field_replace_token,
            max_seq_len=self.max_seq_len,
            ice_template=ice_template,
            prompt_template=prompt_template)

        ds_reader = retriever.dataset_reader
        gold_ans = None
        if ds_reader.output_column:
            gold_ans = ds_reader.dataset['test'][ds_reader.output_column]

        total_samples = len(prompt_list)
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

        entries = [prompt_list[i] for i in todo]
        if gold_ans is not None:
            golds = [gold_ans[i] for i in todo]
        else:
            golds = [None for _ in range(len(entries))]

        logger.info('Starting parallel inference process...')

        extra_gen_kwargs = {}
        sig = inspect.signature(self.model.generate)
        if 'stopping_criteria' in sig.parameters:
            extra_gen_kwargs['stopping_criteria'] = self.stopping_criteria
        if 'min_out_len' in sig.parameters:
            extra_gen_kwargs['min_out_len'] = self.min_out_len

        num_return_sequences = getattr(self.model, 'generation_kwargs',
                                       {}).get('num_return_sequences', 1)

        start_time_stamp = time.time()
        num_sample = 0
        max_workers = self._resolve_max_workers()

        def _infer_one(entry, gold, idx):
            parsed_entry = self.model.parse_template(entry, mode='gen')
            generated = self.model.generate_from_template(
                [entry], max_out_len=self.max_out_len, **extra_gen_kwargs)

            if num_return_sequences == 1:
                prediction = generated[0]
            else:
                prediction = list(batched(generated,
                                          num_return_sequences))[0]

            if self.dump_res_length:
                input_length = 0
                if isinstance(parsed_entry, str):
                    input_length = self.model.get_token_len(parsed_entry)
                elif isinstance(parsed_entry, list):
                    for i in range(len(parsed_entry)):
                        parsed_entry[i]['input_length'] = self.model.get_token_len(
                            parsed_entry[i]['prompt'])
                        input_length += parsed_entry[i]['input_length']

                pred_str = copy.deepcopy(prediction)
                if isinstance(pred_str, dict):
                    pred_str = pred_str['prediction']

                if num_return_sequences == 1:
                    res_length = self.model.get_token_len(pred_str)
                else:
                    res_length = [
                        self.model.get_token_len(pred) for pred in pred_str
                    ]
                return parsed_entry, prediction, idx, gold, res_length, input_length
            return parsed_entry, prediction, idx, gold, None, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_infer_one, entry, gold, idx)
                for idx, entry, gold in zip(todo, entries, golds)
            ]

            completed = total_samples - len(todo)
            for future in as_completed(futures):
                parsed_entry, prediction, idx, gold, res_length, input_length = future.result(
                )
                if self.dump_res_length:
                    output_handler.save_results(parsed_entry,
                                                prediction,
                                                idx,
                                                gold=gold,
                                                res_length=res_length,
                                                input_length=input_length)
                else:
                    output_handler.save_results(parsed_entry,
                                                prediction,
                                                idx,
                                                gold=gold)
                completed += 1
                self._progress_update(1)
                num_sample += 1

                if (self.save_every is not None
                        and completed % self.save_every == 0
                        and self.is_main_process):
                    output_handler.write_to_json(output_json_filepath,
                                                 'tmp_' + output_json_filename)

        end_time_stamp = time.time()

        if self.is_main_process:
            os.makedirs(output_json_filepath, exist_ok=True)
            output_handler.write_to_json(output_json_filepath,
                                         output_json_filename)
            if osp.exists(tmp_json_filepath):
                os.remove(tmp_json_filepath)

        if self.dump_timer and self.is_main_process:
            timer_filepath = os.path.join(output_json_filepath, 'timer',
                                          'time.jsonl')
            os.makedirs(os.path.dirname(timer_filepath), exist_ok=True)
            time_dict = {
                'dataset_name': output_json_filename.removesuffix('.json'),
                'time': end_time_stamp - start_time_stamp,
                'num_sample': num_sample
            }
            with open(timer_filepath, 'a') as f:
                f.write(json.dumps(time_dict) + '\n')

        return [
            sample['prediction']
            for sample in output_handler.results_dict.values()
        ]
