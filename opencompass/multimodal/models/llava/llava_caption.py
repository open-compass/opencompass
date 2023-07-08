import torch
import torch.nn as nn
from mmengine.device import get_device
from transformers import (AutoConfig, AutoTokenizer, CLIPImageProcessor,
                          StoppingCriteria)

from mm_benchmark.registry import MODELS

from .conversation import SeparatorStyle, conv_templates
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


@MODELS.register_module('llava-7b-caption')
class LLaVACaption(nn.Module):
    def __init__(self, model_path) -> None:
        super().__init__()
        self.device, self.dtype = get_device(), torch.float16
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        patch_config(model_path)
        model = LlavaLlamaForCausalLM.from_pretrained(model_path,
                                                      torch_dtype=self.dtype)
        image_processor = CLIPImageProcessor.from_pretrained(
            model.config.mm_vision_tower, torch_dtype=self.dtype)

        mm_use_im_start_end = False  # getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
                special_tokens=True)

        vision_tower = model.model.vision_tower[0]
        vision_tower.to(device=self.device, dtype=self.dtype)
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            (
                vision_config.im_start_token,
                vision_config.im_end_token,
            ) = tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        image_token_len = (vision_config.image_size //
                           vision_config.patch_size)**2

        if 'v1' in model_path.lower():
            self.conv_mode = 'llava_v1'
        elif 'mpt' in model_path.lower():
            self.conv_mode = 'mpt_multimodal'
        else:
            self.conv_mode = 'multimodal'
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mm_use_im_start_end = mm_use_im_start_end
        self.image_token_len = image_token_len  # 256

    def generate(self, batch):
        images = [image.unsqueeze(0) for image in batch['inputs']]
        data_samples = [data_sample for data_sample in batch['data_samples']]
        images = torch.cat(images, dim=0).to(self.device)
        inputs = {'image': images, 'data_samples': data_samples}
        image = inputs.pop('image')
        data_samples = inputs['data_samples']
        samples = {'image': image}

        data_sample = data_samples[0]
        if self.device is not None and 'cuda' in self.device:
            self.model = self.model.to(self.device)
        else:
            device = 'cpu'

        qs = 'a photo of'
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
        # inputs = self.tokenizer([prompt] * len(data_samples))
        inputs = self.tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).to(self.device)

        if image is not None:
            images = image.to(self.device)
        else:
            images = None

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        # TODO (bli): multiple images would raise error in this part, only support single image inference in one GPU for now.
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer,
                                                     input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images.half(),
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

        data_sample.pred_caption = output_text
        return data_sample
