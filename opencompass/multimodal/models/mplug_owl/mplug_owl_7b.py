import os
import sys

import mmengine
import torch
import torch.nn as nn
from mmengine.device import get_device

from opencompass.registry import MM_MODELS


def load_package():
    """Load required packages from llama_adapter_v2_multimodal7b."""
    current_file_path = os.path.abspath(__file__)
    current_folder_path = os.path.dirname(current_file_path)

    sys.path.append(os.path.join(current_folder_path, 'mPLUG-Owl'))  # noqa
    from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
    from mplug_owl.processing_mplug_owl import (MplugOwlImageProcessor,
                                                MplugOwlProcessor)
    from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
    sys.path.pop(-1)

    return MplugOwlForConditionalGeneration, MplugOwlImageProcessor, MplugOwlProcessor, MplugOwlTokenizer  # noqa


MplugOwlForConditionalGeneration, MplugOwlImageProcessor, MplugOwlProcessor, MplugOwlTokenizer = load_package(  # noqa
)  # noqa


@MM_MODELS.register_module('mplug_owl_7b')
class MplugOwl(nn.Module):

    def __init__(self,
                 prompt_constructor: dict,
                 post_processor: dict,
                 model_path='MAGAer13/mplug-owl-llama-7b',
                 mode: str = 'generation'):
        super().__init__()
        pretrained_ckpt = model_path
        # import pdb;pdb.set_trace()
        print(pretrained_ckpt)
        self.model = MplugOwlForConditionalGeneration.from_pretrained(
            pretrained_ckpt,
            torch_dtype=torch.bfloat16,
        ).cuda()
        self.image_processor = MplugOwlImageProcessor.from_pretrained(
            pretrained_ckpt)
        self.tokenizer = MplugOwlTokenizer.from_pretrained(pretrained_ckpt)
        self.processor = MplugOwlProcessor(self.image_processor,
                                           self.tokenizer)
        self.generate_kwargs = {
            'do_sample': False,
            'top_k': 5,
            'max_length': 20,
            'num_beams': 3,
        }

        self.prompt_constructor = mmengine.registry.build_from_cfg(
            prompt_constructor, MM_MODELS)
        if post_processor is not None:
            self.post_processor = mmengine.registry.build_from_cfg(
                post_processor, MM_MODELS)

        self.mode = mode

    def forward(self, batch):
        if self.mode == 'generation':
            return self.generate(batch)

    def generate(self, batch):
        images = [image.unsqueeze(0) for image in batch['inputs']]
        data_samples = [data_sample for data_sample in batch['data_samples']]
        images = torch.cat(images, dim=0).to(get_device())
        inputs = {'image': images, 'data_samples': data_samples}
        inputs = self.prompt_constructor(inputs)
        image = inputs['image']
        prompt = inputs['prompt'][0]
        data_samples = inputs['data_samples']

        data_sample = data_samples[0]
        owl_template = """The following is a conversation
        between a curious human and AI assistant.
        The assistant gives helpful, detailed, and
        polite answers to the user's questions.
        Human: <image>
        Human: {text_input}
        AI: """
        prompt = owl_template.format(text_input=prompt)
        inputs = self.processor(text=[prompt], return_tensors='pt')
        inputs['pixel_values'] = image
        # inputs['pixel_values'] = torch.zeros_like(samples['image'])
        inputs = {
            k: v.bfloat16() if v.dtype == torch.float else v
            for k, v in inputs.items()
        }
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            res = self.model.generate(**inputs, **self.generate_kwargs)
        output_text = self.tokenizer.decode(res.tolist()[0],
                                            skip_special_tokens=True)
        output_text = self.post_processor(output_text)
        data_sample.pred_answer = output_text
        return data_sample
