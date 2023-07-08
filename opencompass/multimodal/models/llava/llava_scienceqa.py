import torch
import torch.nn as nn
from mmengine.device import get_device
from transformers import (AutoConfig, AutoTokenizer, CLIPImageProcessor,
                          StoppingCriteria)

from mm_benchmark.registry import MODELS

from .conversation import SeparatorStyle, conv_templates
from .llava_vqav2 import LLaVAVQAV2
from .model import LlavaLlamaForCausalLM

DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
DEFAULT_IM_START_TOKEN = '<im_start>'
DEFAULT_IM_END_TOKEN = '<im_end>'


def patch_config(config):
    patch_dict = {
        'use_mm_proj': True,
        'mm_vision_tower': 'openai/clip-vit-large-patch14',
        'mm_hidden_size': 1024,
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, 'mm_vision_tower'):
        print(
            f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.'
        )
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


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


@MODELS.register_module('llava-7b-scienceqa')
class LLaVAScienceQA(LLaVAVQAV2):
    choice_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}

    def generate(self, batch):
        images = [image.unsqueeze(0) for image in batch['inputs']]
        data_samples = [data_sample for data_sample in batch['data_samples']]
        images = torch.cat(images, dim=0).to(self.device)
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

        formatted_prompts = []
        for qs in prompts:
            if self.mm_use_im_start_end:
                qs = (qs + '\n' + DEFAULT_IM_START_TOKEN +
                      DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len +
                      DEFAULT_IM_END_TOKEN)
            else:
                qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len

            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            formatted_prompts.append(prompt)

        inputs = self.tokenizer(formatted_prompts)
        input_ids = torch.as_tensor(inputs.input_ids).to(self.device)

        if image is not None:
            image = image.to(self.device)
        else:
            image = None

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        # TODO (bli): multiple images would raise error in this part, only support single image inference in one GPU for now.
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer,
                                                     input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image.half(),
                do_sample=True,
                temperature=0.7,
                max_new_tokens=16,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids !=
                               output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids'
            )
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:],
                                              skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        output_text = outputs.strip()

        data_samples[0].pred_answer = output_text
        return data_samples[0]
