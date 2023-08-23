import re

import mmengine
import torch
import torch.nn as nn

from opencompass.registry import MM_MODELS

from .Otter.models.otter.modeling_otter import OtterForConditionalGeneration


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
    """

    def __init__(self, model_path, load_bit, prompt_constructor) -> None:
        super().__init__()
        torch_dtype = torch.bfloat16 if load_bit == 'bf16' else torch.float32
        self.model = OtterForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype)
        self.tokenizer = self.model.text_tokenizer
        self.tokenizer.padding_side = 'left'
        self.model_dtype = next(self.model.parameters()).dtype
        self.prompt_constructor = mmengine.registry.build_from_cfg(
            prompt_constructor, MM_MODELS)

    def post_process(self, output_text):
        pattern = re.compile(r'([A-Z]\.)')
        res = pattern.findall(output_text)
        if len(res) > 0:
            output_text = res[0][:-1]
        output_text = output_text.strip()
        return output_text

    def generate(self, batch):
        inputs = self.prompt_constructor(batch)
        image = inputs['image']
        prompt = inputs['prompt']
        data_samples = inputs['data_samples']
        data_sample = data_samples[0]
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
        output = self.model.text_tokenizer.decode(generated_text[0])
        output = [x for x in output.split(' ') if not x.startswith('<')]
        try:
            out_label = output.index('GPT:')
        except Exception:
            out_label = output.index('GPT')
        output_text = ' '.join(output[out_label + 1:])

        data_sample.pred_answer = output_text
        return data_sample
