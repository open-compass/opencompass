"""Agent Inferencer."""
import os
import os.path as osp
from typing import List, Optional

import mmengine
from mmengine.registry import Registry
from tqdm import tqdm

from opencompass.registry import ICL_INFERENCERS

from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever
from ..utils.logging import get_logger
from .icl_base_inferencer import BaseInferencer, dump_results_dict

logger = get_logger(__name__)
REGISTRY = Registry('helper')


@ICL_INFERENCERS.register_module()
class AgentInferencer(BaseInferencer):

    def __init__(
            self,
            model,
            output_json_filepath: Optional[str] = './icl_inference_output',
            output_json_filename: Optional[str] = 'predictions',
            save_every: Optional[int] = 1,
            **kwargs) -> None:
        super().__init__(
            model=model,
            output_json_filename=output_json_filename,
            output_json_filepath=output_json_filepath,
            **kwargs,
        )
        self.save_every = save_every

    @property
    def agent(self):
        return self.model

    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None) -> List:
        # 1. Preparation for output logs
        output_handler = AgentInferencerOutputHandler()

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        if 'Fix' in retriever.__class__.__name__:
            ice_idx_list = retriever.retrieve(self.fix_id_list)
        else:
            ice_idx_list = retriever.retrieve()

        # Create tmp json file for saving intermediate results and future
        # resuming
        start = 0
        tmp_json_filepath = os.path.join(output_json_filepath,
                                         'tmp_' + output_json_filename)
        if osp.exists(tmp_json_filepath):
            # TODO: move resume to output handler
            tmp_result_dict = mmengine.load(tmp_json_filepath)
            output_handler.results_dict = tmp_result_dict
            start = len(tmp_result_dict)

        # 3. Inference sample by sample
        logger.info('Starting inference process...')
        for idx, ice_indices in tqdm(enumerate(ice_idx_list[start:], start),
                                     disable=not self.is_main_process):
            user_input = retriever.generate_prompt_for_generate_task(
                idx, ice='', prompt_template=prompt_template)
            gold = retriever.dataset_reader.dataset['test'][
                retriever.dataset_reader.output_column][idx]

            if len(ice_indices) > 0:
                assert ice_template is not None
                ice = [
                    ice_template.generate_ice_item(ice_idx)
                    for ice_idx in ice_indices
                ]
            else:
                ice = None

            answer, steps = self.agent.chat(user_input=user_input, ice=ice)

            # Save current output
            output_handler.save_results(user_input, answer, steps, idx, gold)

            # Save intermediate results
            if (self.save_every is not None and start % self.save_every == 0
                    and self.is_main_process):
                output_handler.write_to_json(output_json_filepath,
                                             'tmp_' + output_json_filename)

        # 4. Output
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


class AgentInferencerOutputHandler:

    def __init__(self) -> None:
        self.results_dict = {}

    def write_to_json(self, save_dir: str, filename: str):
        """Dump the result to a json file."""
        dump_results_dict(self.results_dict, osp.join(save_dir, filename))

    def save_results(self, user_input, answer, steps, idx, gold):
        self.results_dict[str(idx)] = {
            'origin_prompt': user_input,
            'prediction': answer,
            'steps': steps,
            'gold': gold,
        }
