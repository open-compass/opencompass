from typing import Optional

import mmengine
import torch
import torch.nn as nn
from mmengine.device import get_device
from transformers import AutoModel, AutoTokenizer

from opencompass.registry import MM_MODELS


@MM_MODELS.register_module('visualglm')
class VisualGLM(nn.Module):
    """Inference code of VisualGLM.

    We load the visualGLM model via Huggingface.
    Args:
        pretrained_path (str): Path to visualGLM checkpoint or repo id.
        prompt_constructor (dict): The config of prompt constructor.
        post_processor (dict): The config of post processor.
        gen_kwargs (dict): Customize generate function arguments.
    """

    def __init__(self,
                 pretrained_path: str,
                 prompt_constructor: dict,
                 post_processor: dict,
                 gen_kwargs: Optional[dict] = None) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path,
                                                       trust_remote_code=True)
        self.model = AutoModel.from_pretrained(pretrained_path,
                                               trust_remote_code=True).half()
        self.prompt_constructor = mmengine.registry.build_from_cfg(
            prompt_constructor, MM_MODELS)
        self.post_processor = mmengine.registry.build_from_cfg(
            post_processor, MM_MODELS)

        if gen_kwargs:
            self.gen_kwargs = gen_kwargs
        else:
            self.gen_kwargs = dict()

    def encode_by_tokenizer(self, multi_prompts, image_position):
        input_ids = []
        max_seq_length = 0
        for prompt in multi_prompts:
            input0 = self.tokenizer.encode(prompt[:image_position],
                                           add_special_tokens=False)
            input1 = [self.tokenizer.pad_token_id] * self.model.image_length
            input2 = self.tokenizer.encode(prompt[image_position:],
                                           add_special_tokens=False)
            input_all = sum([input0, input1, input2], [])
            input_all = self.tokenizer.build_inputs_with_special_tokens(
                input_all)
            max_seq_length = max(max_seq_length, len(input_all))
            input_ids.append(input_all)
        pre_image_len = len(input0)

        # padding
        for i, _ in enumerate(input_ids):
            pad_len = max_seq_length - len(input_ids[i])
            input_ids[i] = [self.tokenizer.pad_token_id
                            ] * pad_len + input_ids[i]

        return input_ids, pre_image_len

    def generate(self, batch):
        # process input
        image, prompt, data_sample, image_position = self.prompt_constructor(
            batch)
        image = image.to(self.model.dtype).to(get_device())

        # tokenize
        input_all, pre_image_len = self.encode_by_tokenizer(
            prompt, image_position)

        input_all = torch.tensor(input_all, dtype=torch.long).to(get_device())

        # build input param
        inputs = {
            'input_ids': input_all,
            'pre_image_length': pre_image_len,
            'images': image
        }
        # generate answer
        outputs = self.model.generate(**inputs, **self.gen_kwargs)

        # format output
        outputs = outputs.tolist()
        for i, sample in enumerate(data_sample):
            data_sample[i].pred_answer = self.post_processor(
                outputs[i], self.tokenizer, input_all.shape[1])

        return data_sample

    def forward(self, batch):
        return self.generate(batch)
