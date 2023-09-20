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
        is_caption_task (bool): Whether the task is caption task.
            Defaults to False.
        gen_kwargs (dict): Customize generate function arguments.
            Defaults to None.
    """

    def __init__(self,
                 pretrained_path: str,
                 prompt_constructor: dict,
                 post_processor: dict,
                 is_caption_task: bool = False,
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
            self.gen_kwargs = dict(max_length=1024,
                                   min_length=100,
                                   do_sample=True,
                                   temperature=0.8,
                                   top_p=0.4,
                                   top_k=100,
                                   repetition_penalty=1.2)

        self.is_caption_task = is_caption_task

    def encode_by_tokenizer(self, prompt, image_position):

        input0 = self.tokenizer.encode(prompt[:image_position],
                                       add_special_tokens=False)
        input1 = [self.tokenizer.unk_token_id] * self.model.image_length
        input2 = self.tokenizer.encode(prompt[image_position:],
                                       add_special_tokens=False)
        input_all = sum([input0, input1, input2], [])
        input_all = self.tokenizer.build_inputs_with_special_tokens(input_all)
        input_all = torch.tensor(input_all, dtype=torch.long).to(get_device())
        input_all = input_all.unsqueeze(0)

        pre_image_len = len(input0)

        return input_all, pre_image_len

    def generate(self, batch):
        # process input
        image, prompt, data_sample, image_position = self.prompt_constructor(
            batch)
        image = image.to(self.model.dtype).to(get_device())

        # tokenize
        input_all, pre_image_len = self.encode_by_tokenizer(
            prompt, image_position)

        # build input param
        inputs = {
            'input_ids': input_all,
            'pre_image_length': pre_image_len,
            'images': image
        }

        # generate answer
        outputs = self.model.generate(**inputs, **self.gen_kwargs)

        # format output
        outputs = outputs.tolist()[0][input_all.shape[1]:]
        answer = self.post_processor(outputs, self.tokenizer)

        if self.is_caption_task:
            data_sample.pred_caption = answer
        else:
            data_sample.pred_answer = answer

        return data_sample

    def forward(self, batch):
        return self.generate(batch)
