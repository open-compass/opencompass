import sys
import os

import re

import torch
import torch.nn as nn
from mmengine.device import get_device
from transformers import StoppingCriteriaList

from opencompass.registry import MM_MODELS
from .utils import StoppingCriteriaSub


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def load_package():
    """Load required packages from MiniGPT-4."""
    current_file_path = os.path.abspath(__file__)
    current_folder_path = os.path.dirname(current_file_path)

    sys.path.append(os.path.join(current_folder_path, 'MiniGPT-4'))  # noqa
    from minigpt4.models.mini_gpt4 import MiniGPT4

    sys.path.pop(-1)

    return MiniGPT4


MiniGPT4 = load_package()


@MM_MODELS.register_module('minigpt-4-omnimmbench')
class MiniGPT4OmniMMBench(MiniGPT4):
    """Inference code of MiniGPT-4 on OmniMMBench.
    
    Args:
        llama_model (str): The path of vicuna path.
        sys_prompt (str): The prompt added to the begining
            of each query. Defaults to ''.
        low_resource (bool): Whether loaded in low precision.
            Defaults to False.
    """
    def __init__(self,
                 llama_model: str,
                 sys_prompt: str = '',
                 low_resource: bool = False) -> None:
        super().__init__(llama_model=llama_model, low_resource=low_resource)

        cur_device = get_device()
        stop_words_ids = [
            torch.tensor([835]).to(cur_device),
            torch.tensor([2277, 29937]).to(cur_device),
        ]
        self.stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)])
        self.sys_prompt = sys_prompt

    def encode_img(self, image):
        device = image.device

        with self.maybe_autocast():
            image_embeds = self.ln_vision(
                self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1],
                                    dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1,
                                                    -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1],
                                    dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def pack_inputs(self, batch):
        images = [image.unsqueeze(0) for image in batch['inputs']]
        data_samples = [data_sample for data_sample in batch['data_samples']]
        images = torch.cat(images, dim=0).to(get_device())
        inputs = {'image': images, 'data_samples': data_samples}
        return inputs

    def generate(self, batch):
        inputs = self.pack_inputs(batch)
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
        data_sample = data_samples[0]
        img_prompt = '###Human: <Img><ImageHere></Img> '
        if 'context' in samples:
            context_prompt = samples['context'][0]

        question = samples['question']
        options = samples['options']
        if 'context' in samples:
            prompt = img_prompt + ' ' + context_prompt + ' ' + question + ' ' + options  # noqa
        else:
            prompt = img_prompt + ' ' + question + ' ' + options

        # prompt = self.sys_prompt + prompt
        prompt = prompt + '###Assistant:'

        image = samples['image']
        img_embeds, _ = self.encode_img(image)

        prompt_segs = prompt.split('<ImageHere>')
        prompt_seg_tokens = [
            self.llama_tokenizer(seg,
                                 return_tensors='pt',
                                 add_special_tokens=i == 0).
            to(self.llama_model.model.embed_tokens.weight.device).input_ids
            for i, seg in enumerate(prompt_segs)
        ]
        prompt_seg_embs = [
            self.llama_model.model.embed_tokens(seg)
            for seg in prompt_seg_tokens
        ]
        prompt_seg_embs = [prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]]
        prompt_embs = torch.cat(prompt_seg_embs, dim=1)

        # generate output
        outputs = self.llama_model.generate(
            inputs_embeds=prompt_embs,
            max_new_tokens=20,
            num_beams=5,
            do_sample=False,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=-1.0,
            temperature=1.0,
            stopping_criteria=self.stopping_criteria,
            num_return_sequences=1)

        output_token = outputs[0]
        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token,
                                                  add_special_tokens=False)
        output_text = self.post_process(output_text)
        data_sample.pred_answer = output_text
        return data_sample

    def post_process(self, output_text):
        output_text = output_text.split('###')[0]
        output_text = output_text.split('Assistant:')[-1].strip()
        output_text = output_text.strip('</s><s>')
        output_text = output_text.strip('</Img>')
        output_text = output_text.strip()
        pattern = re.compile(r'([A-Z]\.)')
        res = pattern.findall(output_text)
        if len(res) > 0:
            output_text = res[0][:-1]
        return output_text
