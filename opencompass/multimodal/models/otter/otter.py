import importlib

import mmengine
import torch
import torch.nn as nn
from mmengine.device import get_device

from opencompass.registry import MM_MODELS


@MM_MODELS.register_module('otter-9b')
class Otter(nn.Module):
    """Inference code of OTTER.

    Model details:
        OTTER: a multi-modal model based on OpenFlamingo
        (open-sourced version of DeepMind's Flamingo)
        https://github.com/Luodian/Otter
    Args:
        model_path (str): The path of OTTER model
        in Huggingface model hub format.
        load_bit (str): The bit of OTTER model, can be "fp32" or "bf16".
        mode (str): The mode of inference. Defaults to 'generation'.
    """

    def __init__(self,
                 model_path,
                 load_bit,
                 prompt_constructor,
                 post_processor,
                 mode='generation') -> None:
        super().__init__()
        torch_dtype = torch.bfloat16 if load_bit == 'bf16' else torch.float32
        otter_ai = importlib.import_module('otter_ai')
        self.model = otter_ai.OtterForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=get_device())
        self.tokenizer = self.model.text_tokenizer
        self.tokenizer.padding_side = 'left'
        self.model_dtype = next(self.model.parameters()).dtype
        self.prompt_constructor = mmengine.registry.build_from_cfg(
            prompt_constructor, MM_MODELS)
        if post_processor is not None:
            self.post_processor = mmengine.registry.build_from_cfg(
                post_processor, MM_MODELS)
        self.mode = mode

    def forward(self, batch):
        if self.mode == 'generation':
            return self.generate(batch)
        elif self.mode == 'loss':
            return self.loss(batch)
        else:
            raise RuntimeError(f'Invalid mode "{self.mode}".')

    def generate(self, batch):
        inputs = self.prompt_constructor(batch)
        image = inputs['image']
        prompt = inputs['prompt']
        data_samples = inputs['data_samples']
        vision_x = image.unsqueeze(1).unsqueeze(0).to(dtype=self.model_dtype)
        lang_x = self.model.text_tokenizer([prompt], return_tensors='pt')
        bad_words_id = self.model.text_tokenizer(['User:', 'GPT:']).input_ids
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device),
            lang_x=lang_x['input_ids'].to(self.model.device),
            attention_mask=lang_x['attention_mask'].to(self.model.device),
            do_sample=False,
            max_new_tokens=512,
            num_beams=3,
            bad_words_ids=bad_words_id,
            no_repeat_ngram_size=3,
        )
        for i, data_sample in enumerate(data_samples):
            output_text = self.post_processor(generated_text[i],
                                              self.model.text_tokenizer)
            data_sample.pred_answer = output_text
            data_samples[i] = data_sample

        return data_samples
