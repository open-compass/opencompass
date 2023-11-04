import os
import sys

import mmengine
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

    try:
        # the latest version of MiniGPT4
        from minigpt4.models.minigpt4 import MiniGPT4
    except ImportError:
        # the old version of MiniGPT4
        from minigpt4.models.mini_gpt4 import MiniGPT4

    sys.path.pop(-1)

    return MiniGPT4


MiniGPT4 = load_package()


@MM_MODELS.register_module('minigpt-4')
class MiniGPT4Inferencer(MiniGPT4):
    """Inference code of MiniGPT-4.

    Args:
        llama_model (str): The path of vicuna path.
        prompt_constructor (dict): The config of prompt constructor.
        post_processor (dict): The config of post processor.
        do_sample (bool): Whether use sampling. Defaults to False.
        max_length (int): The max length of output. Defaults to 30.
        img_size (int): The size of image. Defaults to 224.
        low_resource (bool): Whether loaded in low precision.
            Defaults to False.
        is_caption_task (bool): Whether the task is caption task.
            Defaults to False.
    """

    def __init__(self,
                 llama_model: str,
                 prompt_constructor: dict,
                 post_processor: dict,
                 do_sample: bool = False,
                 max_length: int = 30,
                 img_size: int = 224,
                 low_resource: bool = False,
                 is_caption_task: bool = False,
                 mode: str = 'generation',
                 n_segments: int = 1) -> None:
        super().__init__(llama_model=llama_model,
                         low_resource=low_resource,
                         img_size=img_size)
        self.mode = mode
        self.n_segments = n_segments

        cur_device = get_device()
        stop_words_ids = [
            torch.tensor([835]).to(cur_device),
            torch.tensor([2277, 29937]).to(cur_device),
        ]
        self.stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)])

        self.prompt_constructor = mmengine.registry.build_from_cfg(
            prompt_constructor, MM_MODELS)
        if post_processor is not None:
            self.post_processor = mmengine.registry.build_from_cfg(
                post_processor, MM_MODELS)
        self.do_sample = do_sample
        self.max_length = max_length
        self.is_caption_task = is_caption_task

    def forward(self, batch):
        if self.mode == 'generation':
            return self.generate(batch)
        elif self.mode == 'loss':
            return self.loss(batch)
        else:
            raise RuntimeError(f'Invalid mode "{self.mode}".')

    def encode_img(self, image):
        device = image.device

        with self.maybe_autocast():
            if image.dim() == 5:
                inputs_llama, atts_llama = [], []
                for j in range(image.size(2)):
                    this_frame = image[:, :, j, :, :]
                    frame_embeds = self.ln_vision(
                        self.visual_encoder(this_frame))
                    frame_atts = torch.ones(frame_embeds.size()[:-1],
                                            dtype=torch.long).to(image.device)

                    query_tokens = self.query_tokens.expand(
                        frame_embeds.shape[0], -1, -1)
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                    frame_inputs_llama = self.llama_proj(
                        frame_query_output.last_hidden_state[:, :query_tokens.
                                                             size(1), :])
                    frame_atts_llama = torch.ones(
                        frame_inputs_llama.size()[:-1],
                        dtype=torch.long).to(image.device)
                    inputs_llama.append(frame_inputs_llama)
                    atts_llama.append(frame_atts_llama)
                inputs_llama = torch.cat(inputs_llama, dim=1)
                atts_llama = torch.cat(atts_llama, dim=1)
            else:
                image_embeds = self.ln_vision(
                    self.visual_encoder(image)).to(device)
                image_atts = torch.ones(image_embeds.size()[:-1],
                                        dtype=torch.long).to(device)

                query_tokens = self.query_tokens.expand(
                    image_embeds.shape[0], -1, -1)
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
        inputs = self.prompt_constructor(inputs)
        image = inputs['image']
        prompt = inputs['prompt']
        data_samples = inputs['data_samples']

        # The main process of generation
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
            max_length=self.max_length,
            num_beams=5,
            do_sample=self.do_sample,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=-1.0,
            temperature=1.0,
            stopping_criteria=self.stopping_criteria,
            num_return_sequences=1)

        for i, data_sample in enumerate(data_samples):
            output_token = outputs[i]
            output_text = self.post_processor(output_token,
                                              self.llama_tokenizer)
            if self.is_caption_task:
                data_sample.pred_caption = output_text
            else:
                data_sample.pred_answer = output_text
            data_samples[i] = data_sample
        return data_samples

    def loss(self, batch):
        inputs = self.pack_inputs(batch)
        inputs = self.prompt_constructor(inputs)
        image = inputs['image']
        batch_size = image.size(0)
        prompt = inputs['prompt']
        data_samples = inputs['data_samples']
        choices = data_samples[0].choices

        with torch.no_grad():
            img_embeds, atts_img = self.encode_img(image)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img,
                                                    prompt)

            self.llama_tokenizer.padding_side = 'right'

            n_cands = len(choices)
            losses = []
            for n in range(self.n_segments):
                seg_len = n_cands // self.n_segments
                if n == (self.n_segments - 1):
                    seg_len = n_cands - seg_len * (self.n_segments - 1)

                to_regress_tokens = self.llama_tokenizer(
                    choices,
                    return_tensors='pt',
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    add_special_tokens=False).to(image.device)

                targets = to_regress_tokens.input_ids.masked_fill(
                    to_regress_tokens.input_ids ==
                    self.llama_tokenizer.pad_token_id, -100)

                empty_targets = (
                    torch.ones([atts_img.shape[0], atts_img.shape[1] + 1],
                               dtype=torch.long).to(image.device).fill_(
                                   -100)  # plus one for bos
                )
                empty_targets = empty_targets.repeat_interleave(seg_len, dim=0)
                targets = torch.cat([empty_targets, targets], dim=1)

                bos = torch.ones([batch_size, 1],
                                 dtype=to_regress_tokens.input_ids.dtype,
                                 device=to_regress_tokens.input_ids.device
                                 ) * self.llama_tokenizer.bos_token_id
                bos_embeds = self.llama_model.model.embed_tokens(bos)
                bos_embeds = bos_embeds.repeat_interleave(seg_len, dim=0)
                img_embeds = img_embeds.repeat_interleave(seg_len, dim=0)

                atts_bos = atts_img[:, :1]
                atts_bos = atts_bos.repeat_interleave(seg_len, dim=0)
                atts_img = atts_img.repeat_interleave(seg_len, dim=0)

                to_regress_embeds = self.llama_model.model.embed_tokens(
                    to_regress_tokens.input_ids)

                inputs_embeds = torch.cat(
                    [bos_embeds, img_embeds, to_regress_embeds], dim=1)
                attention_mask = torch.cat(
                    [atts_bos, atts_img, to_regress_tokens.attention_mask],
                    dim=1)

                with self.maybe_autocast():
                    outputs = self.llama_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        return_dict=True,
                        labels=targets,
                        reduction='none',
                    )
                loss = outputs.loss
                loss = loss.view(targets.size(0), -1).sum(1)
                loss = loss.reshape(batch_size, seg_len)
                losses.append(loss)
            # losses of 4 choices
            losses = torch.cat(losses, dim=-1)[0]

        for i, data_sample in enumerate(data_samples):
            data_sample.losses = losses
            data_samples[i] = data_sample
        return data_samples
