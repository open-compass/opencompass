"""Basic Inferencer."""
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from mmengine.dist import is_main_process
from torch.utils.data import DataLoader

from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever


class BaseInferencer:
    """Base Inferencer class for all evaluation Inferencer.

    Attributes:
        model (:obj:`BaseModel`, optional): The module to inference.
        max_model_token_num (:obj:`int`, optional): Maximum number of
            tokenized words allowed by the LM.
        batch_size (:obj:`int`, optional): Batch size for the
            :obj:`DataLoader`.
        output_json_filepath (:obj:`str`, optional): File path for output
            `JSON` file.
        output_json_filename (:obj:`str`, optional): File name for output
            `JSON` file.
    """
    model = None

    def __init__(
        self,
        model,
        max_seq_len: Optional[int] = None,
        batch_size: Optional[int] = 1,
        output_json_filepath: Optional[str] = './icl_inference_output',
        output_json_filename: Optional[str] = 'predictions',
        fix_id_list: Optional[List[int]] = None,
        **kwargs,
    ) -> None:

        if fix_id_list:
            raise ValueError('Passing fix_id_list to Inferencer is no longer '
                             'allowed. Please pass it to FixKRetriever '
                             'instead.')

        self.model = model

        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.output_json_filepath = output_json_filepath
        self.output_json_filename = output_json_filename
        self.is_main_process = is_main_process()
        os.makedirs(self.output_json_filepath, exist_ok=True)

    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None) -> List:
        """Perform In-Context Inference given a retriever and optional
        templates.

        Args:
            retriever (:obj:`BaseRetriever`): An instance of a Retriever class
                that will be used to retrieve in-context examples
            ice_template (:obj:`PromptTemplate`, optional): A template for
                generating the in-context examples prompt. Defaults to None.
            prompt_template (:obj:`PromptTemplate`, optional): A template for
                generating the final prompt. Defaults to None.
            output_json_filepath (:obj:`str`, optional): The file path to save
                the results as a `JSON` file. Defaults to None.
            output_json_filename (:obj:`str`, optional): The file name to save
                the results as a `JSON` file. Defaults to None.

        Raises:
            NotImplementedError: If the function is not implemented in the
                subclass.

        Returns:
            :obj:`List:` A list of string, each representing the results of one
                inference.
        """
        raise NotImplementedError("Method hasn't been implemented yet")

    @staticmethod
    def get_dataloader(datalist: List[List], batch_size: int) -> DataLoader:
        """Return a dataloader of the input data list."""
        dataloader = DataLoader(datalist,
                                batch_size=batch_size,
                                collate_fn=lambda x: x)
        return dataloader


def dump_results_dict(results_dict, filename):
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(results_dict, json_file, indent=4, ensure_ascii=False)


class GenInferencerOutputHandler:
    origin_prompt_dict = {}
    output_dict = {}
    prediction_dict = {}
    results_dict = {}

    def __init__(self) -> None:
        self.results_dict = {}

    def write_to_json(self, save_dir: str, filename: str):
        """Dump the result to a json file."""
        dump_results_dict(self.results_dict, Path(save_dir) / filename)

    def save_results(self, origin_prompt, prediction, idx, gold=None):
        self.results_dict[str(idx)] = {
            'origin_prompt': origin_prompt,
            'prediction': prediction,
        }
        if gold:
            self.results_dict[str(idx)]['gold'] = gold


class PPLInferencerOutputHandler:
    results_dict = {}

    def __init__(self) -> None:
        self.results_dict = {}

    def write_to_json(self, save_dir: str, filename: str):
        """Dump the result to a json file."""
        dump_results_dict(self.results_dict, Path(save_dir) / filename)

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

    def save_prompt_and_ppl(self, label, input, prompt, ppl, idx):
        if str(idx) not in self.results_dict.keys():
            self.results_dict[str(idx)] = {}
        if 'origin_prompt' not in self.results_dict[str(idx)]:
            self.results_dict[str(idx)]['origin_prompt'] = input
        if 'label: ' + str(label) not in self.results_dict[str(idx)].keys():
            self.results_dict[str(idx)]['label: ' + str(label)] = {}
        self.results_dict[str(idx)]['label: ' +
                                    str(label)]['testing input'] = input
        self.results_dict[str(idx)]['label: ' + str(label)]['prompt'] = prompt
        self.results_dict[str(idx)]['label: ' + str(label)]['PPL'] = ppl

    def save_golds(self, golds):
        for idx, gold in enumerate(golds):
            if str(idx) not in self.results_dict.keys():
                self.results_dict[str(idx)] = {}
            self.results_dict[str(idx)]['gold'] = gold


class CLPInferencerOutputHandler:
    results_dict = {}

    def __init__(self) -> None:
        self.results_dict = {}

    def write_to_json(self, save_dir: str, filename: str):
        """Dump the result to a json file."""
        dump_results_dict(self.results_dict, Path(save_dir) / filename)

    def save_ice(self, ice):
        for idx, example in enumerate(ice):
            if str(idx) not in self.results_dict.keys():
                self.results_dict[str(idx)] = {}
            self.results_dict[str(idx)]['in-context examples'] = example

    def save_prompt_and_condprob(self,
                                 input,
                                 prompt,
                                 cond_prob,
                                 idx,
                                 choices,
                                 gold=None):
        if str(idx) not in self.results_dict.keys():
            self.results_dict[str(idx)] = {}
        # TODO:
        # for single token situation, the input will always be yes currently
        self.results_dict[str(idx)]['testing input'] = input
        self.results_dict[str(idx)]['prompt'] = prompt
        # TODO: hard code here
        self.results_dict[str(idx)]['choices'] = choices
        # For calculate auc scores, set scores as prediction
        self.results_dict[str(idx)]['prediction'] = cond_prob
        # set pred label in case needed
        self.results_dict[str(idx)]['pred_label'] = int(np.argmax(cond_prob))
        self.results_dict[str(idx)]['gold'] = gold
