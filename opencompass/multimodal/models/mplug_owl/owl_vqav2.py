import torch
import torch.nn as nn
from mmengine.device import get_device
from transformers import CLIPImageProcessor

from opencompass.registry import MM_MODELS

# Load via Huggingface Style
from .modeling_mplug_owl import MplugOwlForConditionalGeneration
from .processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from .tokenization_mplug_owl import MplugOwlTokenizer


@MM_MODELS.register_module('mplug_owl-7b-vqav2')
class OwlVQAV2(nn.Module):
    def __init__(self,
                 model_path='MAGAer13/mplug-owl-llama-7b',
                 special_prompt='') -> None:
        super().__init__()
        pretrained_ckpt = model_path
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
            'do_sample': True,
            'top_k': 5,
            'max_length': 16
        }
        self.special_prompt = special_prompt

    def generate(self, batch):
        images = [image.unsqueeze(0) for image in batch['inputs']]
        data_samples = [data_sample for data_sample in batch['data_samples']]
        images = torch.cat(images, dim=0).to(get_device())
        inputs = {'image': images, 'data_samples': data_samples}
        image = inputs.pop('image')
        data_samples = inputs['data_samples']
        samples = {'image': image}

        # data_sample = data_samples[0]
        owl_template = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <image>\nHuman: {question} {special_prompt}\nAI: "
        questions = [
            data_sample.get('question') for data_sample in data_samples
        ]  # noqa
        formatted_questions = []
        for qs in questions:
            formatted_questions.append(
                owl_template.format(question=qs,
                                    special_prompt=self.special_prompt))

        inputs = self.processor(text=formatted_questions, return_tensors='pt')
        inputs['pixel_values'] = samples['image']
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
