"""Sliding Window Cross Entropy Loss Inferencer."""

import math
import os
from typing import List, Optional, Tuple, Union

import mmengine
import numpy as np
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from opencompass.models.base import BaseModel
from opencompass.registry import ICL_INFERENCERS

from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever
from ..utils import get_logger
from .icl_base_inferencer import BaseInferencer, dump_results_dict

logger = get_logger(__name__)


@ICL_INFERENCERS.register_module()
class SWCELossInferencer(BaseInferencer):
    """SWCELossInferencer class to calculate cross entropy loss per batch based
    on a sliding context window approach. This Inferencer is usually used along
    with BPCEvaluator to calculate a models Bits per Character metric on a
    given dataset.

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
        block_size (:obj:`int`, optional): Block size (window size) of
            the sliding window on tokens
        stride (:obj:`int`, optional): Stride (step size) of the
            sliding window on tokens
    """

    def __init__(
            self,
            model: BaseModel,
            max_seq_len: Optional[int] = None,
            batch_size: Optional[int] = 1,
            output_json_filepath: Optional[str] = './icl_inference_output',
            output_json_filename: Optional[str] = 'predictions',
            save_every: Optional[int] = 1,
            block_size: Optional[int] = 1900,
            stride: Optional[int] = 512,
            **kwargs) -> None:
        super().__init__(
            model=model,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            output_json_filename=output_json_filename,
            output_json_filepath=output_json_filepath,
            **kwargs,
        )

        self.block_size = block_size
        self.stride = stride
        self.save_every = save_every
        self.character_num = 0

    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None) -> List:

        # 1. Preparation for output logs
        output_handler = SWCELossInferencerOutputHandler()

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()

        # 3. Generate prompts for testing input
        items_dataset = self.get_encoding_from_retriever_indices(
            ice_idx_list,
            retriever,
            max_seq_len=self.max_seq_len,
            prompt_template=prompt_template)

        # 3-1. Fetch and zip prompt & gold answer if output column exists
        ds_reader = retriever.dataset_reader

        assert ds_reader.output_column is None, (
            'SWCELossInferencer supports `output_column=None` only.')

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
                index = len(
                    tmp_result_dict)  # rewrite tmp_dataset on every run

        # 4. Initialize torch dataset from items hf dataset
        logger.info('Starting dataset building process...')

        eval_dataset = SlidingWindowEvalDataset(
            items_dataset,
            block_size=self.block_size + 1,
            stride=self.stride,
        )

        # 4-1. Construct Dataloader
        dataloader = DataLoader(eval_dataset, self.batch_size, shuffle=False)

        # 5. Calculate total loss in each batch
        logger.info('Starting inference process...')

        device = self.model.model.device
        for ind, datum in enumerate(
                tqdm(dataloader, disable=not self.is_main_process)):

            if ind < index:
                continue

            encodings = datum['input_ids']  # encodings
            attention_mask = datum['attention_mask']

            # 5-1. Loss calculation by local model
            with torch.no_grad():
                if self.batch_size == 1:
                    input_ids = encodings[0:self.block_size].contiguous().to(
                        device)
                    targets = encodings[1:self.block_size +
                                        1].contiguous().long().to(device)
                    attention_mask = attention_mask[1:self.block_size +
                                                    1].contiguous().to(device)
                else:
                    input_ids = encodings[:,
                                          0:self.block_size].contiguous().to(
                                              device)
                    targets = encodings[:, 1:self.block_size +
                                        1].contiguous().long().to(device)
                    attention_mask = attention_mask[:, 1:self.block_size +
                                                    1].contiguous().to(device)

                logits = self.model.model(input_ids).logits
                loss = self._get_cross_entropy(logits,
                                               targets,
                                               attention_mask=attention_mask)
                loss = loss.cpu().item()

                logger.info(f'loss: {loss:.8f}')

            # 5-2. Save intermediate results
            output_handler.save_results(loss, datum['total_chr_num'][0].item(),
                                        index)
            index = index + 1

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

        return [sample for sample in output_handler.results_dict.values()]

    def get_encoding_from_retriever_indices(
            self,
            ice_idx_list: List[List[int]],
            retriever: BaseRetriever,
            max_seq_len: Optional[int] = None,
            prompt_template: Optional[PromptTemplate] = None,
            dtype: str = 'auto') -> Tuple[List, List]:

        vocab_size = self.model.tokenizer.vocab_size

        if dtype == 'auto':
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            if vocab_size is not None and vocab_size < 65500:
                _dtype = np.uint16
            else:
                _dtype = np.int32
        else:
            _dtype = dtype

        item_list = []
        for idx, ice_idx in enumerate(ice_idx_list):
            cur_item_dict = {}

            prompt = retriever.generate_prompt_for_generate_task(
                idx,
                ice='',
                ice_template=None,
                prompt_template=prompt_template)

            cur_item_dict['prompt'] = prompt

            # Get encodings from model tokenizer
            # As long as block_size > max_seq_len, we can safely ignore the
            # warning about token sequence length
            cur_item_dict['encoding'] = np.array(
                self.model.tokenizer.encode(prompt), dtype=_dtype)

            item_list.append(cur_item_dict)

        items = HFDataset.from_list(item_list)

        return items

    def _get_cross_entropy(self,
                           logits: torch.Tensor,
                           targets: torch.Tensor,
                           attention_mask: torch.Tensor = None):
        """Calculate cross entropy based on given logits, targets and
        attention_mask for BPC loss calculation.

        Args:
            logits (np.ndarray): Model logits
            targets (np.ndarray): Targets
            attention_mask (torch.Tensor, optional): Attention mask.
                Defaults to None.

        Returns:
            torch.Tensor: Total cross entropy on the given batch of logits and
            targets reduced by summation
        """
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(-1)
            targets = targets.masked_fill(~attention_mask, -1)

        return torch.nn.functional.cross_entropy(logits,
                                                 targets,
                                                 ignore_index=-1,
                                                 reduction='sum')


class SlidingWindowEvalDataset(Dataset):

    def __init__(self,
                 data: HFDataset,
                 block_size: int = 1900,
                 stride: int = 512) -> None:
        """SlidingWindowEvalDataset.

        Args:
            data (HFDataset): HuggingFace dataset containing input samples
            block_size (int, optional): Sliding context window size.
                Defaults to 1900.
            stride (int, optional): Sliding context window step size.
                Defaults to 512.
        """
        self.block_size = block_size
        self.data = data
        self.stride = stride

        self._prepare()
        self.prev_end_loc = 0
        self.seq_len = len(self.data)
        self.begin_loc = 0

    def _prepare(self):
        """Prepare evaluation dataset by calculating total number of characters
        and from original text and concatenating encodings into a single
        array."""
        self._curr_idx = 0
        self._arr = []

        self._total_chr_num = 0
        for i in range(len(self.data)):
            self._total_chr_num += len(self.data[i]['prompt'])

        logger.info(f'data Dataset before concat: {self.data}')

        self.data = np.concatenate([a['encoding'] for a in self.data], axis=0)

        logger.info(f'data after concat: {self.data}')
        logger.info(f'data after concat: {self.data.shape}')

    def __len__(self):
        return math.floor((len(self.data) - self.block_size) / self.stride + 1)

    def __getitem__(self, item):
        end_loc = min(self.begin_loc + self.block_size, self.seq_len)
        trg_len = end_loc - self.prev_end_loc

        input_ids = self.data[self.begin_loc:end_loc]

        attention_mask = np.ones((len(input_ids), ), dtype=bool)
        attention_mask[:-trg_len] = False

        self.prev_end_loc = end_loc
        self.begin_loc = self.begin_loc + self.stride

        out_items = dict(
            input_ids=torch.tensor(input_ids),
            attention_mask=torch.tensor(attention_mask, dtype=bool),
            total_chr_num=self._total_chr_num,
        )
        return out_items

    @property
    def total_chr_num(self):
        return self._total_chr_num


class SWCELossInferencerOutputHandler:
    origin_prompt_dict = {}
    output_dict = {}
    results_dict = {}

    def __init__(self) -> None:
        self.results_dict = {}

    def write_to_json(self, save_dir: str, filename: str):
        """Dump the result to a json file."""
        dump_results_dict(self.results_dict, os.path.join(save_dir, filename))

    def save_results(self, loss: float, total_chr_num: int,
                     idx: Union[str, int]) -> None:

        self.results_dict[str(idx)] = {
            'loss': loss,
            'total_chr_num': total_chr_num
        }
