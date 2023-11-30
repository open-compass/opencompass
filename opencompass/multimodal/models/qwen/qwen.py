import types
from typing import Optional, Tuple

import mmengine
import torch
import torch.nn as nn
from mmengine.device import get_device
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.modeling_outputs import BaseModelOutputWithPast

from opencompass.registry import MM_MODELS

from .generation_utils import decode_tokens, make_context


@MM_MODELS.register_module('qwen-vl-base')
class QwenVLBase(nn.Module):
    """Inference code of Qwen-VL.

    We load the Qwen model via Huggingface.
    Args:
        pretrained_path (str): Path to Qwen checkpoint or repo id.
        prompt_constructor (dict): The config of prompt constructor.
        post_processor (dict): The config of post processor.
        is_caption_task (bool): Whether the task is caption task.
            Defaults to False.
        commit_id (str): Use given version of Qwen-VL.
            Warning: the latest version may have some conflicts.
            Recommend to use the given default version.
    """

    def __init__(
            self,
            pretrained_path: str,
            prompt_constructor: dict = None,
            post_processor: dict = None,
            is_caption_task: bool = False,
            commit_id: str = '548275c8b99de56dec203c0e793be18e030f2f4c'
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path,
                                                       trust_remote_code=True,
                                                       revision=commit_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            device_map=get_device(),
            trust_remote_code=True,
            revision=commit_id)
        self.model.generation_config = GenerationConfig.from_pretrained(
            pretrained_path, trust_remote_code=True, revision=commit_id)
        if prompt_constructor is not None:
            self.prompt_constructor = mmengine.registry.build_from_cfg(
                prompt_constructor, MM_MODELS)
        if post_processor is not None:
            self.post_processor = mmengine.registry.build_from_cfg(
                post_processor, MM_MODELS)
        else:
            self.post_processor = None
        self.is_caption_task = is_caption_task
        self.model.transformer.forward = types.MethodType(
            forward_hack, self.model.transformer)

    def _build_embeds(self, images, input_ids):
        # encode image
        images = self.model.transformer.visual(images)
        # compute image position
        bos_pos = torch.where(input_ids == self.model.transformer.config.
                              visual['image_start_id'])
        eos_pos = torch.where(
            input_ids ==
            self.model.transformer.config.visual['image_start_id'] + 1)
        assert (bos_pos[0] == eos_pos[0]).all()
        img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
        # embed words
        inputs_embeds = self.model.transformer.wte(input_ids)
        # embed image tokens
        for idx, (i, a, b) in enumerate(img_pos):
            inputs_embeds[i][a + 1:b] = images[idx]
        return inputs_embeds

    def generate(self, batch):
        images = batch.pop('inputs')
        images = torch.stack(images, dim=0)
        format_input = self.prompt_constructor(batch)
        query = self.tokenizer.from_list_format(format_input)

        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = inputs.to(get_device())
        input_ids, token_type_ids, attention_mask = inputs[
            'input_ids'], inputs['token_type_ids'], inputs['attention_mask']
        inputs_embeds = self._build_embeds(images, input_ids)
        pred = self.model.generate(input_ids=input_ids,
                                   inputs_embeds=inputs_embeds,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
        response = self.post_processor(pred.cpu()[0])

        data_sample = batch['data_samples'][0]
        if self.is_caption_task:
            data_sample.pred_caption = response
        else:
            data_sample.pred_answer = response
        return data_sample

    def forward(self, batch):
        return self.generate(batch)


@MM_MODELS.register_module('qwen-vl-chat')
class QwenVLChat(QwenVLBase):
    """Inference code of Qwen-VL-Chat.

    We load the Qwen model via Huggingface.
    Args:
        pretrained_path (str): Path to Qwen checkpoint or repo id.
        prompt_constructor (dict): The config of prompt constructor.
        post_processor (dict): The config of post processor.
        is_caption_task (bool): Whether the task is caption task.
            Defaults to False.
    """

    def __init__(self,
                 pretrained_path: str,
                 prompt_constructor: dict = None,
                 post_processor: dict = None,
                 is_caption_task: bool = False) -> None:
        super().__init__(pretrained_path, prompt_constructor, post_processor,
                         is_caption_task)

    def generate(self, batch):
        images = batch.pop('inputs')
        images = torch.stack(images, dim=0)
        format_input = self.prompt_constructor(batch)
        query = self.tokenizer.from_list_format(format_input)

        raw_text, context_tokens = make_context(
            self.tokenizer,
            query,
            system='You are a helpful assistant.',
            chat_format=self.model.generation_config.chat_format,
        )

        input_ids = torch.tensor([context_tokens]).to(get_device())

        inputs_embeds = self._build_embeds(images, input_ids)
        pred = self.model.generate(input_ids=input_ids,
                                   inputs_embeds=inputs_embeds)

        response = decode_tokens(
            pred[0],
            self.tokenizer,
            raw_text_len=len(raw_text),
            context_length=len(context_tokens),
            chat_format=self.model.generation_config.chat_format,
            verbose=False,
            errors='replace')

        if self.post_processor:
            response = self.post_processor(response)

        data_sample = batch['data_samples'][0]
        if self.is_caption_task:
            data_sample.pred_caption = response
        else:
            data_sample.pred_answer = response
        return data_sample


def forward_hack(self,
                 input_ids: Optional[torch.LongTensor] = None,
                 past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                 attention_mask: Optional[torch.FloatTensor] = None,
                 token_type_ids: Optional[torch.LongTensor] = None,
                 position_ids: Optional[torch.LongTensor] = None,
                 head_mask: Optional[torch.FloatTensor] = None,
                 inputs_embeds: Optional[torch.FloatTensor] = None,
                 encoder_hidden_states: Optional[torch.Tensor] = None,
                 encoder_attention_mask: Optional[torch.FloatTensor] = None,
                 use_cache: Optional[bool] = None,
                 output_attentions: Optional[bool] = None,
                 output_hidden_states: Optional[bool] = None,
                 return_dict: Optional[bool] = None):
    if past_key_values is None and input_ids is not None and torch.any(
            input_ids == self.config.visual['image_start_id']):
        bos_pos = torch.where(
            input_ids == self.config.visual['image_start_id'])
        eos_pos = torch.where(
            input_ids == self.config.visual['image_start_id'] + 1)
        assert (bos_pos[0] == eos_pos[0]).all()
        img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
        images = []
        for i, a, b in img_pos:
            image = input_ids[i][a + 1:b - 1].tolist()
            image = image[:image.index(self.config.visual['image_start_id'] +
                                       2)]
            images.append(bytes(image).decode('utf-8'))

        images = self.visual.encode(images)
        assert images.shape[0] == len(images)
    else:
        images = None

    output_attentions = (output_attentions if output_attentions is not None
                         else self.config.output_attentions)
    output_hidden_states = (output_hidden_states if output_hidden_states
                            is not None else self.config.output_hidden_states)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (return_dict
                   if return_dict is not None else self.config.use_return_dict)

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            'You cannot specify both input_ids and inputs_embeds at the same time'  # noqa
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError(
            'You have to specify either input_ids or inputs_embeds')

    device = input_ids.device if input_ids is not None else inputs_embeds.device  # noqa

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])
    if position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
    else:
        past_length = past_key_values[0][0].size(-2)

    if position_ids is None:
        position_ids = torch.arange(
            past_length,
            input_shape[-1] + past_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    encoder_attention_mask = None
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)

    if batch_size <= 0:
        raise ValueError('batch_size has to be defined and > 0')
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_length)

    hidden_states = inputs_embeds

    hidden_states = self.drop(hidden_states)
    if images is not None:
        for idx, (i, a, b) in enumerate(img_pos):
            hidden_states[i][a + 1:b] = images[idx]
    output_shape = input_shape + (hidden_states.size(-1), )

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None
    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):

                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache, output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                None,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[2 if output_attentions else 1], )

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[1], )

    hidden_states = self.ln_f(hidden_states)
    hidden_states = hidden_states.view(output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states, )

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states]
                     if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )
