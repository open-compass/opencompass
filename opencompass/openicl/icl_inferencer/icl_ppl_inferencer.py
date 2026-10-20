# flake8: noqa
# yapf: disable
"""PPL Inferencer."""

import os
from typing import List, Optional

import torch
from tqdm import trange

from opencompass.models.base import BaseModel
from opencompass.registry import ICL_INFERENCERS

from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever
from ..utils import get_logger
from .icl_base_inferencer import BaseInferencer, PPLInferencerOutputHandler

logger = get_logger(__name__)


@ICL_INFERENCERS.register_module()
class PPLInferencer(BaseInferencer):
    """PPL Inferencer class to evaluate by perplexity.

    Attributes:
        model (:obj:`BaseModel`, optional): The module to inference.
        max_seq_len (:obj:`int`): Maximum number of tokenized words allowed by
            the LM.
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`
        output_json_filepath (:obj:`str`, optional): File path for output
            `JSON` file.
        output_json_filename (:obj:`str`, optional): File name for output
            `JSON` file.
        labels (:obj:`List`, optional): A list of labels for all classes.
    """

    def __init__(
            self,
            model: BaseModel,
            max_seq_len: Optional[int] = None,
            batch_size: Optional[int] = 1,
            output_json_filepath: Optional[str] = './icl_inference_output',
            output_json_filename: Optional[str] = 'predictions',
            labels: Optional[List] = None,
            **kwargs) -> None:
        super().__init__(
            model=model,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            output_json_filename=output_json_filename,
            output_json_filepath=output_json_filepath,
            **kwargs,
        )

        self.labels = labels

    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None,
                  normalizing_str: Optional[str] = None) -> List:
        # 1. Preparation for output logs
        output_handler = PPLInferencerOutputHandler()

        sub_predictions = []
        ppl = []
        ice = []

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()

        # 3. Get labels of all the classes
        if self.labels is None:
            labels = retriever.get_labels(ice_template=ice_template,
                                          prompt_template=prompt_template)
        else:
            labels = self.labels

        # 4. Generate in-context examples for testing inputs
        for idx in range(len(ice_idx_list)):
            ice.append(retriever.generate_ice(ice_idx_list[idx], ice_template=ice_template))
        output_handler.save_ice(self.model.parse_template(ice, mode='ppl'))

        # 5. Calculating PPL for prompts in each label's class
        for label in labels:
            index = 0
            prompt_list = []
            sub_ppl_list = []
            token_num_list = []
            normalizing_prompt_list = []
            context_length_list = []

            # 5.1 Generate prompts of current label and truncate
            # TODO: Refactor
            for idx in range(len(ice_idx_list)):
                prompt_kwargs = {
                    'idx': idx,
                    'ice': ice[idx],
                    'label': label,
                    'ice_template': ice_template,
                    'prompt_template': prompt_template,
                    'remain_sep': normalizing_str is not None
                }
                prompt = retriever.generate_label_prompt(**prompt_kwargs)
                prompt_token_num = self.model.get_token_len_from_template(prompt, mode='ppl')
                if self.max_seq_len is not None:
                    while len(ice_idx_list[idx]) > 0 and prompt_token_num > self.max_seq_len:
                        ice_idx_list[idx] = ice_idx_list[idx][:-1]
                        ice[idx] = retriever.generate_ice(ice_idx_list[idx], ice_template=ice_template)
                        prompt_kwargs['ice'] = ice[idx]
                        prompt = retriever.generate_label_prompt(**prompt_kwargs)
                        prompt_token_num = self.model.get_token_len_from_template(prompt, mode='ppl')

                if normalizing_str is not None:
                    assert isinstance(prompt, str), 'Prompt must be a string when normalizing_str is set.'
                    prompt_sep = prompt
                    if prompt_template is not None:
                        sep_token = prompt_template.sep_token
                    else:
                        sep_token = ice_template.sep_token
                    sep_pos = prompt_sep.find(sep_token)

                    context = prompt_sep[0:sep_pos]
                    answer = prompt_sep[sep_pos:].replace(sep_token, '')
                    prompt = context + answer
                    normalizing_prompt = normalizing_str + answer

                    context_length_list.append(self.model.get_token_len_from_template(context, mode='ppl'))
                    normalizing_prompt_list.append(normalizing_prompt)

                prompt_list.append(prompt)
                token_num_list.append(prompt_token_num)

            if normalizing_str is not None:
                normalizing_str_len = self.model.get_token_len_from_template(
                    normalizing_str, mode='ppl')

            # 5.2 Get PPL
            logger.info(f"Calculating PPL for prompts labeled '{label}'")
            for idx in trange(0, len(prompt_list), self.batch_size, disable=not self.is_main_process):
                sub_prompt_list = prompt_list[idx:idx + self.batch_size]
                with torch.no_grad():
                    if normalizing_str is not None:
                        sub_context_length_list = context_length_list[idx:idx + self.batch_size]
                        sub_normalizing_prompt_list = normalizing_prompt_list[idx:idx + self.batch_size]
                        res1 = self.model.get_ppl_from_template(sub_prompt_list, mask_length=sub_context_length_list)
                        sub_normalizing_context_length_list = [normalizing_str_len for _ in range(len(sub_prompt_list))]
                        res2 = self.model.get_ppl_from_template(sub_normalizing_prompt_list, mask_length=sub_normalizing_context_length_list)
                        sub_res = res1 - res2
                    else:
                        sub_res = self.model.get_ppl_from_template(sub_prompt_list).tolist()

                for res, prompt in zip(sub_res, self.model.parse_template(sub_prompt_list, mode='ppl')):
                    sub_ppl_list.append(res)
                    ice_str = self.model.parse_template(ice[idx], mode='ppl')
                    prompt_wo_ice = prompt.replace(ice_str, '')
                    output_handler.save_prompt_and_ppl(label, prompt_wo_ice, prompt, res, index)
                    output_handler.results_dict[str(index)][f'label: {str(label)}']['BPB'] = res * token_num_list[index] / len(prompt_wo_ice.encode())
                    index = index + 1
            ppl.append(sub_ppl_list)

        # 6. Get lowest PPL class as predictions
        ppl = list(zip(*ppl))
        for single_ppl in ppl:
            sub_predictions.append(labels[single_ppl.index(min(single_ppl))])
        output_handler.save_predictions(sub_predictions)

        # 7. Fetch gold answers if exist
        ds_reader = retriever.dataset_reader
        if ds_reader.output_column:
            golds = ds_reader.dataset['test'][ds_reader.output_column]
            output_handler.save_golds(golds)

        # 8. Output
        if self.is_main_process:
            os.makedirs(output_json_filepath, exist_ok=True)
            output_handler.write_to_json(output_json_filepath, output_json_filename)

        return [sample['prediction'] for sample in output_handler.results_dict.values()]
