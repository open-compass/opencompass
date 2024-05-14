# flake8: noqa
# yapf: disable
"""LogLikelihood(LL) Inferencer."""

import os
from typing import List, Optional

import torch
from tqdm import trange

from opencompass.models.base import BaseModel
from opencompass.registry import ICL_INFERENCERS

from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever
from ..utils import get_logger
from .icl_base_inferencer import BaseInferencer, dump_results_dict

logger = get_logger(__name__)


@ICL_INFERENCERS.register_module()
class LLInferencer(BaseInferencer):
    """Loglikelihood Inferencer class to evaluate by loglikelihood.

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
                  output_json_filename: Optional[str] = None) -> List:
        # 1. Preparation for output logs
        output_handler = LLInferencerOutputHandler()

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
            labels = retriever.get_labels(ice_template=ice_template, prompt_template=prompt_template)
        else:
            labels = self.labels

        # 4. Generate in-context examples for testing inputs
        for idx in range(len(ice_idx_list)):
            ice.append(retriever.generate_ice(ice_idx_list[idx], ice_template=ice_template))
        output_handler.save_ice(self.model.parse_template(ice, mode='ppl'))

        # 5. Calculating loglikelihood for prompts in each label's class
        for label in labels:
            index = 0
            prompt_list = []
            sub_ppl_list = []
            token_num_list = []
            cont_list = []

            # 5.1 Generate prompts of current label and truncate
            # TODO: Refactor
            for idx in range(len(ice_idx_list)):
                prompt_kwargs = {
                    'idx': idx,
                    'ice': ice[idx],
                    'label': label,
                    'ice_template': ice_template,
                    'prompt_template': prompt_template,
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

                prompt_list.append(prompt)
                token_num_list.append(prompt_token_num)
                cont_list.append(retriever.test_ds[idx]['cont'])

            # 5.2 Get loglikelihood
            logger.info(f"Calculating Loglikelihood for prompts labeled '{label}'")
            for idx in trange(0, len(prompt_list), self.batch_size, disable=not self.is_main_process):
                sub_prompt_list = prompt_list[idx:idx + self.batch_size]
                sub_cont_list = cont_list[idx:idx + self.batch_size]

                with torch.no_grad():
                    # mainly modify compared to PPLInferencer
                    sub_inputs = self.model.parse_template(sub_prompt_list, mode='ppl')
                    sub_res = self.model.get_loglikelihood(sub_inputs, sub_cont_list).tolist()
                for res, prompt in zip(sub_res, self.model.parse_template(sub_prompt_list, mode='ppl')):
                    sub_ppl_list.append(res)
                    ice_str = self.model.parse_template(ice[idx], mode='ppl')
                    output_handler.save_prompt_and_loglikelihood(label, prompt.replace(ice_str, ''), prompt, res, index)
                    index = index + 1
            ppl.append(sub_ppl_list)

        # 6. Get lowest PPL class as predictions
        ppl = list(zip(*ppl))
        for single_ppl in ppl:
            sub_predictions.append(labels[single_ppl.index(max(single_ppl))])
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


class LLInferencerOutputHandler:
    results_dict = {}

    def __init__(self) -> None:
        self.results_dict = {}

    def write_to_json(self, save_dir: str, filename: str):
        """Dump the result to a json file."""
        dump_results_dict(self.results_dict, os.path.join(save_dir, filename))

    def save_ice(self, ice):
        for idx, example in enumerate(ice):
            if str(idx) not in self.results_dict.keys():
                self.results_dict[str(idx)] = {}
            self.results_dict[str(idx)]['in-context examples'] = example

    def save_predictions(self, predictions):
        for idx, prediction in enumerate(predictions):
            if str(idx) not in self.results_dict.keys():
                self.results_dict[str(idx)] = {}
            self.results_dict[str(idx)]['prediction'] = prediction

    def save_prompt_and_loglikelihood(self, label, input, prompt,
                                      loglikelihood, idx):
        if str(idx) not in self.results_dict.keys():
            self.results_dict[str(idx)] = {}
        if 'label: ' + str(label) not in self.results_dict[str(idx)].keys():
            self.results_dict[str(idx)]['label: ' + str(label)] = {}
        self.results_dict[str(idx)]['label: ' +
                                    str(label)]['testing input'] = input
        self.results_dict[str(idx)]['label: ' + str(label)]['prompt'] = prompt
        self.results_dict[str(idx)][
            'label: ' + str(label)]['Loglikelihood'] = loglikelihood

    def save_golds(self, golds):
        for idx, gold in enumerate(golds):
            if str(idx) not in self.results_dict.keys():
                self.results_dict[str(idx)] = {}
            self.results_dict[str(idx)]['gold'] = gold
