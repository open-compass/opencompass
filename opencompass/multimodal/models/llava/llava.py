import importlib
import os
import sys

import torch
import torch.nn as nn
from mmengine.device import get_device
from transformers import StoppingCriteria

from opencompass.registry import MM_MODELS

from .prompt_constructor import LLaVAMMBenchPromptConstructor


def load_package():
    """Load required packages from LLaVA."""
    current_file_path = os.path.abspath(__file__)
    current_folder_path = os.path.dirname(current_file_path)

    sys.path.append(os.path.join(current_folder_path, 'LLaVA'))  # noqa
    return


class KeywordsStoppingCriteria(StoppingCriteria):

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


@MM_MODELS.register_module('llava-7b-mmbench')
class LLaVA(nn.Module):
    """Inference code of LLaVA on MMBench. Need to clone LLaVA official repo
    first. Please check out the README in config.

    Args:
        model_path (str): The path of llava checkpoint.
    """

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.device, self.dtype = get_device(), torch.float16

        # load LLaVA modules
        load_package()
        mm_utils = importlib.import_module('llava.mm_utils')
        builder = importlib.import_module('llava.model.builder')
        conversation = importlib.import_module('llava.conversation')
        self.SeparatorStyle = conversation.SeparatorStyle
        self.conv_templates = conversation.conv_templates

        # load pretrained LLaVA
        model_name = mm_utils.get_model_name_from_path(model_path)
        tokenizer, model, _, _ = builder.load_pretrained_model(
            model_path, None, model_name)

        # load prompt constructor and post processor
        if 'v1' in model_path.lower():
            conv_mode = 'llava_v1'
        elif 'mpt' in model_path.lower():
            conv_mode = 'mpt_multimodal'
        else:
            conv_mode = 'multimodal'
        mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end',
                                      False)

        self.model = model
        self.tokenizer = tokenizer
        self.prompt_constructor = LLaVAMMBenchPromptConstructor(
            conv_templates=conversation.conv_templates,
            conv_mode=conv_mode,
            image_token_len=256,
            mm_use_im_start_end=mm_use_im_start_end)

    def generate(self, batch):

        prompt, stop_str = self.prompt_constructor(batch)
        keywords = [stop_str]

        inputs = self.tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).to(self.device)
        data_sample = batch['data_samples'][0]

        self.model = self.model.to(self.device)
        image = batch['inputs'][0].unsqueeze(0)
        if image is not None:
            images = image.to(self.device)
        else:
            images = None

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
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        output_text = outputs.strip()

        data_sample.pred_answer = output_text
        return data_sample
