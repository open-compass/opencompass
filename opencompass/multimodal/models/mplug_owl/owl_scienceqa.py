import torch
import torch.nn as nn
from mmengine.device import get_device
from transformers import CLIPImageProcessor

from opencompass.registry import MM_MODELS

# Load via Huggingface Style
from .modeling_mplug_owl import MplugOwlForConditionalGeneration
from .owl_vqav2 import OwlVQAV2
from .processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from .tokenization_mplug_owl import MplugOwlTokenizer


@MM_MODELS.register_module('mplug_owl-7b-scienceqa')
class OwlScienceQA(OwlVQAV2):
    choice_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}

    def generate(self, batch):
        images = [image.unsqueeze(0) for image in batch['inputs']]
        data_samples = [data_sample for data_sample in batch['data_samples']]
        images = torch.cat(images, dim=0).to(get_device())
        inputs = {'image': images, 'data_samples': data_samples}
        image = inputs.pop('image')
        data_samples = inputs['data_samples']

        questions = [
            'Question: ' + data_sample.get('question') + '\n'
            for data_sample in data_samples
        ]  # noqa
        choices = [data_sample.get('choices') for data_sample in data_samples]
        choices = [[
            f'({self.choice_mapping[i]}) ' + item
            for i, item in enumerate(choice)
        ] for choice in choices]
        choices = [
            'Choices: ' + ' '.join(choice) + '\n' for choice in choices
        ]  # noqa
        contexts = [
            'Context: ' + data_sample.get('hint') + '\n'
            for data_sample in data_samples
        ]  # noqa

        prompts = [
            context + question + choice
            for context, question, choice in zip(contexts, questions, choices)
        ]

        # data_sample = data_samples[0]
        owl_template = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <image>\nHuman: {question}\nAI: "

        formatted_questions = []
        for qs in prompts:
            formatted_questions.append(owl_template.format(question=qs))

        inputs = self.processor(text=formatted_questions, return_tensors='pt')
        inputs['pixel_values'] = image
        inputs = {
            k: v.bfloat16() if v.dtype == torch.float else v
            for k, v in inputs.items()
        }
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.generate_kwargs)

        for idx, res in enumerate(outputs):
            output_text = self.tokenizer.decode(res.tolist(),
                                                skip_special_tokens=True)
            data_samples[idx].pred_answer = output_text
        return data_samples
