import re

import torch
import torch.nn as nn
from mmengine.device import get_device
from transformers import CLIPImageProcessor

from opencompass.registry import MM_MODELS

# Load via Huggingface Style
from .modeling_mplug_owl import MplugOwlForConditionalGeneration
from .processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from .tokenization_mplug_owl import MplugOwlTokenizer


@MM_MODELS.register_module('mplug_owl-7b-mm-benchmark')
class Owl(nn.Module):
    def __init__(self, model_path='MAGAer13/mplug-owl-llama-7b') -> None:
        super().__init__()
        pretrained_ckpt = model_path
        # import pdb;pdb.set_trace()
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

    def post_process(self, output_text):
        pattern = re.compile(r'([A-Z]\.)')
        res = pattern.findall(output_text)
        if len(res) > 0:
            output_text = res[0][:-1]
        return output_text

    def generate(self, batch):
        images = [image.unsqueeze(0) for image in batch['inputs']]
        data_samples = [data_sample for data_sample in batch['data_samples']]
        images = torch.cat(images, dim=0).to(get_device())
        inputs = {'image': images, 'data_samples': data_samples}
        image = inputs.pop('image')
        data_samples = inputs['data_samples']
        samples = {'image': image}
        question = [
            data_sample.get('question') for data_sample in data_samples
        ]
        options = [data_sample.get('options') for data_sample in data_samples]
        samples.update({'question': question[0]})
        samples.update({'options': options[0]})
        if data_samples[0].get('context') is not None:
            context = [
                data_sample.get('context') for data_sample in data_samples
            ]
            samples.update({'context': context})

        if 'context' in samples:
            context_prompt = samples['context'][0]

        question = samples['question']
        options = samples['options']
        if 'context' in samples:
            prompt = context_prompt + ' ' + question + ' ' + options  # noqa
        else:
            prompt = question + ' ' + options

        data_sample = data_samples[0]
        owl_template = """The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
        Human: <image>
        Human: {text_input}
        AI: """
        prompt = owl_template.format(text_input=prompt)
        inputs = self.processor(text=[prompt], return_tensors='pt')
        inputs['pixel_values'] = samples['image']
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
        output_text = self.post_process(output_text)
        data_sample.pred_answer = output_text
        return data_sample
