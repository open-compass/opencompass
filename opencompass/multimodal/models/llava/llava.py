import importlib
import os
import sys

import mmengine
import torch
import torch.nn as nn
from mmengine.device import get_device
from transformers import StoppingCriteria

from opencompass.registry import MM_MODELS

IMAGE_TOKEN_INDEX = -200


def load_package():
    """Load required packages from LLaVA."""
    current_file_path = os.path.abspath(__file__)
    current_folder_path = os.path.dirname(current_file_path)

    sys.path.append(os.path.join(current_folder_path, 'LLaVA'))  # noqa
    return


class KeywordsStoppingCriteria(StoppingCriteria):
    """Keyword stopping criteria implemented for llava."""

    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor,
                 **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:,
                                                             self.start_len:],
                                                  skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


@MM_MODELS.register_module('llava')
class LLaVA(nn.Module):
    """Inference code of LLaVA. Need to clone LLaVA official repo first. Please
    check out the README in config.

    Args:
        model_path (str): The path of llava checkpoint.
        prompt_constructor (dict): The config of prompt constructor.
        post_processor (dict): The config of post processor.
        is_caption_task (bool): Whether the task is caption task.
            Defaults to False.
    """

    def __init__(
        self,
        model_path: str,
        prompt_constructor: dict,
        post_processor: dict,
        is_caption_task: bool = False,
    ) -> None:
        super().__init__()
        self.dtype = torch.float16
        self.is_caption_task = is_caption_task

        # load LLaVA modules
        load_package()
        mm_utils = importlib.import_module('llava.mm_utils')
        builder = importlib.import_module('llava.model.builder')

        # load pretrained LLaVA
        # Note: When encounters with device related errors,
        # try setting `low_cpu_mem_usage` in `load_pretrained_model` as False
        model_name = mm_utils.get_model_name_from_path(model_path)
        tokenizer, model, _, _ = builder.load_pretrained_model(
            model_path, None, model_name)
        vision_tower = model.get_vision_tower()
        vision_tower.to(device=get_device(), dtype=self.dtype)
        model.to(device=get_device(), dtype=self.dtype)

        # load prompt constructor and post processor
        if 'v1' in model_path.lower():
            conv_mode = 'llava_v1'
        elif 'mpt' in model_path.lower():
            conv_mode = 'mpt_multimodal'
        else:
            conv_mode = 'multimodal'
        mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end',
                                      False)
        prompt_constructor.update({
            'conv_mode': conv_mode,
            'mm_use_im_start_end': mm_use_im_start_end
        })
        self.prompt_constructor = mmengine.registry.build_from_cfg(
            prompt_constructor, MM_MODELS)
        self.post_processor = mmengine.registry.build_from_cfg(
            post_processor, MM_MODELS)
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, batch):

        prompt, stop_str = self.prompt_constructor(batch)
        keywords = [stop_str]
        data_sample = batch['data_samples'][0]

        image = batch['inputs'][0].unsqueeze(0)
        if image is not None:
            images = image.to(get_device())
        else:
            images = None

        mm_utils = importlib.import_module('llava.mm_utils')
        input_ids = mm_utils.tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
            return_tensors='pt').unsqueeze(0).to(get_device())

        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer,
                                                     input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images.half(),
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids !=
                               output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids'  # noqa
            )
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:],
                                              skip_special_tokens=True)[0]

        output_text = self.post_processor(outputs, stop_str)

        if self.is_caption_task:
            data_sample.pred_caption = output_text
        else:
            data_sample.pred_answer = output_text
        return data_sample

    def forward(self, batch):
        return self.generate(batch)
