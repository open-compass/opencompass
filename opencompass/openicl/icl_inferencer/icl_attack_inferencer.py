"""Direct Generation Inferencer."""

import os
import os.path as osp
from typing import List, Optional

import mmengine
import torch
from tqdm import tqdm

from opencompass.models.base import BaseModel
from opencompass.registry import (ICL_EVALUATORS, ICL_INFERENCERS,
                                  TEXT_POSTPROCESSORS)

from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever
from ..utils.logging import get_logger
from .icl_base_inferencer import BaseInferencer, GenInferencerOutputHandler

logger = get_logger(__name__)


@ICL_INFERENCERS.register_module()
class AttackInferencer(BaseInferencer):
    """Generation Inferencer class to directly evaluate by generation.

    Attributes:
        model (:obj:`BaseModelWrapper`, optional): The module to inference.
        max_out_len (:obj:`int`, optional): Maximum number of tokenized words
            of the output.
        adv_key (:obj:`str`): Prompt key in template to be attacked.
        metric_key (:obj:`str`): Metric key to be returned and compared.
            Defaults to `accuracy`.
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
    """

    def __init__(
            self,
            model: BaseModel,
            max_out_len: int,
            adv_key: str,
            metric_key: str = 'accuracy',
            max_seq_len: Optional[int] = None,
            batch_size: Optional[int] = 1,
            gen_field_replace_token: Optional[str] = '',
            output_json_filepath: Optional[str] = './icl_inference_output',
            output_json_filename: Optional[str] = 'predictions',
            save_every: Optional[int] = 1,
            dataset_cfg: Optional[List[int]] = None,
            **kwargs) -> None:
        super().__init__(
            model=model,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            output_json_filename=output_json_filename,
            output_json_filepath=output_json_filepath,
            **kwargs,
        )

        self.adv_key = adv_key
        self.metric_key = metric_key
        self.dataset_cfg = dataset_cfg
        self.eval_cfg = dataset_cfg['eval_cfg']
        self.output_column = dataset_cfg['reader_cfg']['output_column']
        self.gen_field_replace_token = gen_field_replace_token
        self.max_out_len = max_out_len

        if self.model.is_api and save_every is None:
            save_every = 1
        self.save_every = save_every

    def predict(self, adv_prompt) -> List:
        # 1. Preparation for output logs
        output_handler = GenInferencerOutputHandler()

        # if output_json_filepath is None:
        output_json_filepath = self.output_json_filepath
        # if output_json_filename is None:
        output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = self.retriever.retrieve()

        # 3. Generate prompts for testing input
        prompt_list, label_list = self.get_generation_prompt_list_from_retriever_indices(  # noqa
            ice_idx_list, {self.adv_key: adv_prompt},
            self.retriever,
            self.gen_field_replace_token,
            max_seq_len=self.max_seq_len,
            ice_template=self.ice_template,
            prompt_template=self.prompt_template)

        # 3.1 Fetch and zip prompt & gold answer if output column exists
        ds_reader = self.retriever.dataset_reader
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
        logger.info('Starting inference process...')
        for datum in tqdm(dataloader, disable=not self.is_main_process):
            if ds_reader.output_column:
                entry, golds = list(zip(*datum))
            else:
                entry = datum
                golds = [None for _ in range(len(entry))]
            # 5-1. Inference with local model
            with torch.no_grad():
                parsed_entries = self.model.parse_template(entry, mode='gen')
                results = self.model.generate_from_template(
                    entry, max_out_len=self.max_out_len)
                generated = results

            # 5-3. Save current output
            for prompt, prediction, gold in zip(parsed_entries, generated,
                                                golds):
                output_handler.save_results(prompt,
                                            prediction,
                                            index,
                                            gold=gold)
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
            if osp.exists(tmp_json_filepath):
                os.remove(tmp_json_filepath)

        pred_strs = [
            sample['prediction']
            for sample in output_handler.results_dict.values()
        ]

        if 'pred_postprocessor' in self.eval_cfg:
            kwargs = self.eval_cfg['pred_postprocessor'].copy()
            proc = TEXT_POSTPROCESSORS.get(kwargs.pop('type'))
            pred_strs = [proc(s, **kwargs) for s in pred_strs]

        icl_evaluator = ICL_EVALUATORS.build(self.eval_cfg['evaluator'])
        result = icl_evaluator.score(predictions=pred_strs,
                                     references=label_list)
        score = result.get(self.metric_key)
        # try to shrink score to range 0-1
        return score / 100 if score > 1 else score

    def get_generation_prompt_list_from_retriever_indices(
            self,
            ice_idx_list: List[List[int]],
            extra_prompt: dict,
            retriever: BaseRetriever,
            gen_field_replace_token: str,
            max_seq_len: Optional[int] = None,
            ice_template: Optional[PromptTemplate] = None,
            prompt_template: Optional[PromptTemplate] = None):
        prompt_list = []
        label_list = []
        for idx, ice_idx in enumerate(ice_idx_list):
            ice = retriever.generate_ice(ice_idx, ice_template=ice_template)
            prompt = retriever.generate_prompt_for_adv_generate_task(
                idx,
                ice,
                extra_prompt,
                gen_field_replace_token=gen_field_replace_token,
                ice_template=ice_template,
                prompt_template=prompt_template)
            label = retriever.test_ds[idx][self.output_column]
            label_list.append(label)
            if max_seq_len is not None:
                prompt_token_num = self.model.get_token_len_from_template(
                    prompt, mode='gen')
                while len(ice_idx) > 0 and prompt_token_num > max_seq_len:
                    ice_idx = ice_idx[:-1]
                    ice = retriever.generate_ice(ice_idx,
                                                 ice_template=ice_template)
                    prompt = retriever.generate_prompt_for_adv_generate_task(
                        idx,
                        ice,
                        extra_prompt,
                        gen_field_replace_token=gen_field_replace_token,
                        ice_template=ice_template,
                        prompt_template=prompt_template)
                    prompt_token_num = self.model.get_token_len_from_template(
                        prompt, mode='gen')
            prompt_list.append(prompt)
        return prompt_list, label_list
