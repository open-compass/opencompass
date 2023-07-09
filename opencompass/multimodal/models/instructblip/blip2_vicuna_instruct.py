"""Requires Transformer 4.28 and above, implementation may change according the
Llama implementation."""
import logging

import mmengine
import torch
import torch.nn as nn
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from mmengine.device import get_device
from transformers import LlamaForCausalLM, LlamaTokenizer

from opencompass.registry import MM_MODELS


@MM_MODELS.register_module('blip2-vicuna-instruct')
class InstructBlipInferencer(Blip2Base):

    def __init__(
        self,
        prompt_constructor: dict,
        post_processor: dict,
        vit_model: str = 'eva_clip_g',
        img_size: int = 224,
        drop_path_rate: float = 0,
        use_grad_checkpoint: bool = False,
        vit_precision: str = 'fp16',
        freeze_vit: bool = True,
        num_query_token: int = 32,
        llm_model: str = '',
        sys_prompt: str = '',
        prompt: str = '',
        max_txt_len: int = 128,
        max_output_txt_len: int = 256,
        qformer_text_input: bool = True,
        low_resource: bool = False,
        mode: str = 'generation',
        is_caption_task=False,
    ):
        super().__init__()
        self.mode = mode
        self.prompt_constructor = mmengine.registry.build_from_cfg(
            prompt_constructor, MM_MODELS)
        self.post_processor = mmengine.registry.build_from_cfg(
            post_processor, MM_MODELS)

        self.tokenizer = self.init_tokenizer(truncation_side='left')

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint,
            vit_precision)
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info('freeze vision encoder')

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features)

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        self.llm_tokenizer = LlamaTokenizer.from_pretrained(
            llm_model, use_fast=False, truncation_side='left')

        if low_resource:
            self.llm_model = LlamaForCausalLM.from_pretrained(
                llm_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': 0})
        else:
            self.llm_model = LlamaForCausalLM.from_pretrained(
                llm_model, torch_dtype=torch.float16)
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.llm_proj = nn.Linear(self.Qformer.config.hidden_size,
                                  self.llm_model.config.hidden_size)

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.sys_prompt = sys_prompt
        self.prompt = prompt
        self.is_caption_task = is_caption_task

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input

    def forward(self, batch):
        if self.mode == 'generation':
            return self.generate(batch)
        else:
            raise RuntimeError(f'Invalid mode "{self.mode}".')

    def concat_text_input_output(self, input_ids, input_atts, output_ids,
                                 output_atts):
        input_part_targets_len = []
        llm_tokens = {'input_ids': [], 'attention_mask': []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones], output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ]))
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones], output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ]))
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(
            llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def pack_inputs(self, batch):
        images = [image.unsqueeze(0) for image in batch['inputs']]
        data_samples = [data_sample for data_sample in batch['data_samples']]
        images = torch.cat(images, dim=0).to(get_device())
        inputs = {'image': images, 'data_samples': data_samples}
        return inputs

    @torch.no_grad()
    def generate(
        self,
        batch,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        inputs = self.pack_inputs(batch)
        inputs = self.prompt_constructor(inputs)
        image = inputs['image']
        prompt = inputs['prompt']
        data_samples = inputs['data_samples']

        self.llm_tokenizer.padding_side = 'left'

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(
                prompt
            ) == bs, 'The number of prompts must be equal to the batch size.'

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors='pt',
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1],
                                    dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],
                                     dim=1)

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        if self.qformer_text_input:
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_llm = self.llm_proj(
            query_output.last_hidden_state[:, :query_tokens.size(1), :])
        atts_llm = torch.ones(inputs_llm.size()[:-1],
                              dtype=torch.long).to(image.device)

        prompt = ['###Human: ' + p + '###Assistant:' for p in prompt]
        prompt = [self.sys_prompt + p for p in prompt]
        llm_tokens = self.llm_tokenizer(prompt,
                                        padding='longest',
                                        return_tensors='pt').to(image.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(
                llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask],
                                       dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=self.max_output_txt_len,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        for i, data_sample in enumerate(data_samples):
            output_token = outputs[i]
            output_text = self.post_processor(output_token, self.llm_tokenizer)
            if self.is_caption_task:
                data_sample.pred_caption = output_text
            else:
                data_sample.pred_answer = output_text
            data_samples[i] = data_sample
        return data_samples
