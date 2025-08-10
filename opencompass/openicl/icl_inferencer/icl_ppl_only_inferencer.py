"""PPL Inferencer."""

import os
from typing import List, Optional

import mmengine
import torch
from tqdm import tqdm

from opencompass.models.base import BaseModel
from opencompass.registry import ICL_INFERENCERS

from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever
from ..utils import get_logger
from .icl_base_inferencer import BaseInferencer, dump_results_dict

logger = get_logger(__name__)


@ICL_INFERENCERS.register_module()
class PPLOnlyInferencer(BaseInferencer):
    """PPLOnlyInferencer class to calculate PPL and PPL only, no choice is
    made. This Inferencer is usually used along with AveragePPLEvaluator.

    Attributes:
        model (:obj:`BaseModel`, optional): The module to inference.
        max_seq_len (:obj:`int`): Maximum number of tokenized words allowed by
            the LM.
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`
        output_json_filepath (:obj:`str`, optional): File path for output
            `JSON` file.
        output_json_filename (:obj:`str`, optional): File name for output
            `JSON` file.
        save_every (:obj:`int`, optional): Save intermediate results every
    """

    def __init__(
            self,
            model: BaseModel,
            max_seq_len: Optional[int] = None,
            batch_size: Optional[int] = 1,
            output_json_filepath: Optional[str] = './icl_inference_output',
            output_json_filename: Optional[str] = 'predictions',
            save_every: Optional[int] = 1,
            **kwargs) -> None:
        super().__init__(
            model=model,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            output_json_filename=output_json_filename,
            output_json_filepath=output_json_filepath,
            **kwargs,
        )

        self.save_every = save_every

    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None) -> List:
        # 1. Preparation for output logs
        output_handler = PPLOnlyInferencerOutputHandler()

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()

        # 3. Generate prompts for testing input
        prompt_list = self.get_generation_prompt_list_from_retriever_indices(
            ice_idx_list,
            retriever,
            max_seq_len=self.max_seq_len,
            ice_template=ice_template,
            prompt_template=prompt_template)

        # 3.1 Fetch and zip prompt & gold answer if output column exists
        ds_reader = retriever.dataset_reader

        assert ds_reader.output_column is None, (
            'PPLOnlyInferencer supports `output_column=None` only.')

        # Create tmp json file for saving intermediate results and future
        # resuming
        index = 0
        tmp_json_filepath = os.path.join(output_json_filepath,
                                         'tmp_' + output_json_filename)
        if os.path.exists(tmp_json_filepath):
            # TODO: move resume to output handler
            try:
                tmp_result_dict = mmengine.load(tmp_json_filepath)
            except Exception:
                pass
            else:
                output_handler.results_dict = tmp_result_dict
                index = len(tmp_result_dict)

        # 4. Wrap prompts with Dataloader
        dataloader = self.get_dataloader(prompt_list[index:], self.batch_size)

        # 5. Inference for prompts in each batch
        logger.info('Starting inference process...')
        for datum in tqdm(dataloader, disable=not self.is_main_process):
            entry = datum
            # 5-1. Inference with local model
            with torch.no_grad():
                ppls = self.model.get_ppl_from_template(entry).tolist()

            parsed_entries = self.model.parse_template(entry, mode='gen')
            # 5-3. Save current output
            for prompt, ppl, in zip(parsed_entries, ppls):
                output_handler.save_results(prompt, ppl, index)
                index = index + 1

            # 5-4. Save intermediate results
            if (self.save_every is not None and index % self.save_every == 0
                    and self.is_main_process):
                output_handler.write_to_json(output_json_filepath,
                                             'tmp_' + output_json_filename)

        # 6. Output
        if self.is_main_process:
            os.makedirs(output_json_filepath, exist_ok=True)
            output_handler.write_to_json(output_json_filepath,
                                         output_json_filename)
            if os.path.exists(tmp_json_filepath):
                os.remove(tmp_json_filepath)

        return [
            sample['ppl'] for sample in output_handler.results_dict.values()
        ]

    def get_generation_prompt_list_from_retriever_indices(
            self,
            ice_idx_list: List[List[int]],
            retriever: BaseRetriever,
            max_seq_len: Optional[int] = None,
            ice_template: Optional[PromptTemplate] = None,
            prompt_template: Optional[PromptTemplate] = None):
        prompt_list = []
        for idx, ice_idx in enumerate(ice_idx_list):
            ice = retriever.generate_ice(ice_idx, ice_template=ice_template)
            prompt = retriever.generate_prompt_for_generate_task(
                idx,
                ice,
                ice_template=ice_template,
                prompt_template=prompt_template)
            if max_seq_len is not None:
                prompt_token_num = self.model.get_token_len_from_template(
                    prompt, mode='gen')
                while len(ice_idx) > 0 and prompt_token_num > max_seq_len:
                    ice_idx = ice_idx[:-1]
                    ice = retriever.generate_ice(ice_idx,
                                                 ice_template=ice_template)
                    prompt = retriever.generate_prompt_for_generate_task(
                        idx,
                        ice,
                        ice_template=ice_template,
                        prompt_template=prompt_template)
                    prompt_token_num = self.model.get_token_len_from_template(
                        prompt, mode='gen')
            prompt_list.append(prompt)
        return prompt_list


class PPLOnlyInferencerOutputHandler:
    origin_prompt_dict = {}
    output_dict = {}
    results_dict = {}

    def __init__(self) -> None:
        self.results_dict = {}

    def write_to_json(self, save_dir: str, filename: str):
        """Dump the result to a json file."""
        dump_results_dict(self.results_dict, os.path.join(save_dir, filename))

    def save_results(self, origin_prompt, ppl, idx):
        self.results_dict[str(idx)] = {
            'origin_prompt': origin_prompt,
            'ppl': ppl,
        }
