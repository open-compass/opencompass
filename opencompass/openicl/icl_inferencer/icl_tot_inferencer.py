"""Tree-of-Thought Generation Inferencer."""

import itertools
import os
import os.path as osp
from typing import List, Optional

import mmengine
import numpy as np
import torch
from tqdm import tqdm

from opencompass.models.base import BaseModel
from opencompass.registry import ICL_INFERENCERS, TOT_WRAPPER

from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever
from ..utils.logging import get_logger
from .icl_gen_inferencer import GenInferencer, GenInferencerOutputHandler

logger = get_logger(__name__)


@ICL_INFERENCERS.register_module()
class ToTInferencer(GenInferencer):
    """Tree-of-Thought Inferencer class to evaluate by tree style reasoning
    paths.
    Doc: https://opencompass.readthedocs.io/en/latest/prompt/
         chain_of_thought.html
    Official tot paper: https://arxiv.org/pdf/2305.10601.pdf


    Attributes:
        model (:obj:`BaseModelWrapper`, optional): The module to inference.
        max_seq_len (:obj:`int`, optional): Maximum number of tokenized words
            allowed by the LM.
        batch_size (:obj:`int`, optional): Batch size for the
            :obj:`DataLoader`.
        output_json_filepath (:obj:`str`, optional): File path for output
            `JSON` file.
        output_json_filename (:obj:`str`, optional): File name for output
            `JSON` file.
        gen_field_replace_token (:obj:`str`, optional): Used to replace the
            generation field token when generating prompts.
        save_every (:obj:`int`, optional): Save intermediate results every
            `save_every` iters. Defaults to 1.
        generation_kwargs (:obj:`Dict`, optional): Parameters for the
            :obj:`model.generate()` method.
        naive_run (:obj:`bool`): if True, run naive IO/CoT sampling instead of
            ToT + BFS.
        prompt_wrapper (:obj:`dict`): wrapper for prompts
        prompt_sample (:obj:`str`): (choices=[standard, cot]) sampling prompt
        method_generate (:obj:`str`): (choices=[sample, propose])
            thought generator,whether to sample independent thoughts (used in
            Creative Writing task) or propose sequential thoughts (used in Game
            of 24)
        method_evaluate (:obj:`str`): (choices=[value, vote]) state evaluator,
            whether to use the value states independently (used in Game of 24)
            or vote on states together (used in Creative Writing)
        n_generate_sample (:obj:`int`): number of times to prompt for
            thought generation
        n_evaluate_sample(:obj:`int`): number of times to prompt for
            state evaluation
        n_select_sample (:obj:`int`): number of states to keep from each step
            (i.e. b in the Tree-of-Thought paper's ToT + BFS algorithm)
    """

    def __init__(
            self,
            model: BaseModel,
            max_out_len: int,
            max_seq_len: Optional[int] = None,
            batch_size: Optional[int] = 1,
            gen_field_replace_token: Optional[str] = '',
            output_json_filepath: Optional[str] = './icl_inference_output',
            output_json_filename: Optional[str] = 'predictions',
            save_every: Optional[int] = 1,
            naive_run: bool = False,
            prompt_wrapper: dict = {},
            prompt_sample: str = 'standard',
            method_generate: str = 'sample',
            method_evaluate: str = 'value',
            method_select: str = 'greedy',
            n_generate_sample: int = 1,
            n_evaluate_sample: int = 1,
            n_select_sample: int = 1,
            generation_kwargs: dict = {},
            **kwargs) -> None:
        super().__init__(
            model=model,
            max_out_len=max_out_len,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            gen_field_replace_token=gen_field_replace_token,
            output_json_filename=output_json_filename,
            output_json_filepath=output_json_filepath,
            save_every=save_every,
            sc_size=n_evaluate_sample,
            **kwargs,
        )
        self.max_out_len = max_out_len
        self.prompt_wrapper = TOT_WRAPPER.build(prompt_wrapper)
        self.naive_run = naive_run
        self.prompt_sample = prompt_sample
        self.method_generate = method_generate
        self.method_evaluate = method_evaluate
        self.method_select = method_select
        self.n_generate_sample = n_generate_sample
        self.n_evaluate_sample = n_evaluate_sample
        self.n_select_sample = n_select_sample
        self.generation_kwargs = generation_kwargs

    def get_value(self,
                  x: str,
                  y: str,
                  n_evaluate_sample: int,
                  cache_value: bool = True) -> str:
        """Get evaluation value of a partial output.

        Args:
            x (str): The input text to be evaluated.
            y (str): The partial output to be evaluated.
            n_evaluate_sample (int): Times to evaluate each partial output.
            cache_value (bool): Cache to avoid duplicate candidates.
                Defaults to True.
        Returns:
            str: Value of evaluated partial outputs.
        """
        value_prompt = self.prompt_wrapper.value_prompt_wrap(x, y)
        if cache_value and value_prompt in self.prompt_wrapper.value_cache:
            return self.prompt_wrapper.value_cache[value_prompt]
        value_outputs = self.model.generate_from_template(
            [value_prompt],
            max_out_len=self.max_out_len,
            num_beams=n_evaluate_sample,
            num_return_sequences=n_evaluate_sample,
            **self.generation_kwargs)
        value = self.prompt_wrapper.value_outputs_unwrap(x, y, value_outputs)
        if cache_value:
            self.prompt_wrapper.value_cache[value_prompt] = value
        return value

    def get_values(self,
                   x: str,
                   ys: List[str],
                   n_evaluate_sample: int,
                   cache_value: bool = True) -> List[str]:
        """Get evaluation values of partial outputs.

        Args:
            x (str): The input text to be solved.
            ys (List[str]): The partial outputs to be evaluated.
            n_evaluate_sample (int): Times to evaluate each partial output.
            cache_value (bool): Cache to avoid duplicate candidates.
                Defaults to True.

        Returns:
            List[str]: Values of evaluated partial outputs.
        """
        values = []
        local_value_cache = {}
        for y in ys:  # each partial output
            if y in local_value_cache:  # avoid duplicate candidates
                value = 0
            else:
                value = self.get_value(x,
                                       y,
                                       n_evaluate_sample,
                                       cache_value=cache_value)
                local_value_cache[y] = value
            values.append(value)
        return values

    def get_votes(self, x: str, ys: List[str],
                  n_evaluate_sample: int) -> List[str]:
        """Get votes of partial outputs.

        Args:
            x (str): The input text to be solved.
            ys (List[str]): The partial outputs to be evaluated.
            n_evaluate_sample (int): Times to evaluate each partial output.

        Returns:
            List[str]: Values of evaluated partial outputs.
        """
        vote_prompt = self.prompt_wrapper.vote_prompt_wrap(x, ys)
        vote_outputs = self.model.generate_from_template(
            [vote_prompt],
            max_out_len=self.max_out_len,
            num_beams=n_evaluate_sample,
            num_return_sequences=n_evaluate_sample,
            **self.generation_kwargs)
        values = self.prompt_wrapper.vote_outputs_unwrap(vote_outputs, len(ys))
        return values

    def get_proposals(self, x: str, y: str) -> List[str]:
        """Get proposal prompts.

        Args:
            x (str): The input text to be solved.
            y (str): The partial output.

        Returns:
            List[str]: Proposal prompts.
        """
        propose_prompt = self.prompt_wrapper.propose_prompt_wrap(x, y)
        proposals = self.model.generate_from_template(
            [propose_prompt],
            max_out_len=self.max_out_len,
            num_beams=1,
            num_return_sequences=1,
            **self.generation_kwargs)[0].split('\n')
        return [y + _ + '\n' for _ in proposals]

    def get_samples(self, x: str, y: str, n_generate_sample: int,
                    prompt_sample: str):
        """Get samples from a partial output.

        Args:
            x (str): The input text to be solved.
            y (str): The partial output.
            n_generate_sample (int): Times to generate samples.
            prompt_sample (str): (choices=[standard, cot]) sampling prompt

        Returns:
            List[str]: Samples from a partial output.
        """
        if prompt_sample == 'standard':
            prompt = self.prompt_wrapper.standard_prompt_wrap(x, y)
        elif prompt_sample == 'cot':
            prompt = self.prompt_wrapper.cot_prompt_wrap(x, y)
        else:
            raise ValueError(f'prompt_sample {prompt_sample} not recognized')
        samples = self.model.generate_from_template(
            [prompt],
            max_out_len=self.max_out_len,
            num_beams=n_generate_sample,
            num_return_sequences=n_generate_sample,
            **self.generation_kwargs)
        return [y + _ for _ in samples]

    def tot_solve(self, x: str) -> str:
        """Solve a problem using Tree-of-Thought algorithm.

        Args:
            x (str): The input text to be solved.

        Returns:
            str: Final answer of the problem.
        """
        ys = ['']  # current output candidates
        infos = []
        for step in range(self.prompt_wrapper.steps):
            logger.info(f'\n-- step {str(step)} --\n')
            # generation
            if self.method_generate == 'sample':
                new_ys = [
                    self.get_samples(x,
                                     y,
                                     self.n_generate_sample,
                                     prompt_sample=self.prompt_sample)
                    for y in ys
                ]
            elif self.method_generate == 'propose':
                new_ys = [self.get_proposals(x, y) for y in ys]
            new_ys = list(itertools.chain(*new_ys))
            ids = list(range(len(new_ys)))
            # evaluation
            if self.method_evaluate == 'vote':
                values = self.get_votes(x, new_ys, self.n_evaluate_sample)
            elif self.method_evaluate == 'value':
                values = self.get_values(x, new_ys, self.n_evaluate_sample)

            # selection
            if self.method_select == 'sample':
                ps = np.array(values) / sum(values)
                select_ids = np.random.choice(ids,
                                              size=self.n_select_sample,
                                              p=ps).tolist()
            elif self.method_select == 'greedy':
                select_ids = sorted(ids, key=lambda x: values[x],
                                    reverse=True)[:self.n_select_sample]
            select_new_ys = [new_ys[select_id] for select_id in select_ids]

            # log
            sorted_new_ys, sorted_values = zip(
                *sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            logger.info(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: '
                        f'{sorted_values}\n-- choices --: {select_new_ys}\n')

            infos.append({
                'step': step,
                'x': x,
                'ys': ys,
                'new_ys': new_ys,
                'values': values,
                'select_new_ys': select_new_ys
            })
            ys = select_new_ys
            logger.info(ys)

        return ys

    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None) -> List:
        # 1. Preparation for output logs
        output_handler = GenInferencerOutputHandler()

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
            self.gen_field_replace_token,
            max_seq_len=self.max_seq_len,
            ice_template=ice_template,
            prompt_template=prompt_template)

        # 3.1 Fetch and zip prompt & gold answer if output column exists
        ds_reader = retriever.dataset_reader
        if ds_reader.output_column:
            gold_ans = ds_reader.dataset['test'][ds_reader.output_column]
            prompt_list = list(zip(prompt_list, gold_ans))

        # Create tmp json file for saving intermediate results and future
        # resuming
        index = 0
        tmp_json_filepath = os.path.join(output_json_filepath,
                                         'tmp_' + output_json_filename)
        if osp.exists(tmp_json_filepath):
            # TODO: move resume to output handler
            tmp_result_dict = mmengine.load(tmp_json_filepath)
            output_handler.results_dict = tmp_result_dict
            index = len(tmp_result_dict)

        # 4. Wrap prompts with Dataloader
        dataloader = self.get_dataloader(prompt_list[index:], self.batch_size)

        # 5. Inference for prompts in each batch
        logger.info('Starting ToT inference process...')
        for datum in tqdm(dataloader, disable=not self.is_main_process):
            if ds_reader.output_column:
                entries, golds = list(zip(*datum))
            else:
                entries = datum
                golds = [None for _ in range(len(entries))]
            # 5-1. Inference with ToT and local model
            with torch.no_grad():
                parsed_entries = self.model.parse_template(entries, mode='gen')
                generated = [self.tot_solve(entry) for entry in entries]

            # 5-2. Save current output
            for prompt, prediction, gold in zip(parsed_entries, generated,
                                                golds):
                output_handler.save_results(prompt,
                                            prediction,
                                            index,
                                            gold=gold)
                index = index + 1

            # 5-3. Save intermediate results
            if (self.save_every is not None and index % self.save_every == 0
                    and self.is_main_process):
                output_handler.write_to_json(output_json_filepath,
                                             'tmp_' + output_json_filename)

        # 6. Output
        if self.is_main_process:
            os.makedirs(output_json_filepath, exist_ok=True)
            output_handler.write_to_json(output_json_filepath,
                                         output_json_filename)
            if osp.exists(tmp_json_filepath):
                os.remove(tmp_json_filepath)

        return [
            sample['prediction']
            for sample in output_handler.results_dict.values()
        ]
