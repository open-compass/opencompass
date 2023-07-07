import contextlib
import logging
import os
import re

import torch
import torch.nn as nn
from mmengine.device import get_device
from transformers import BertTokenizer, LlamaTokenizer, StoppingCriteriaList

from opencompass.registry import MM_MODELS

from .eva_vit import create_eva_vit_g
from .modelling_llama import LlamaForCausalLM
from .Qformer import BertConfig, BertLMHeadModel
from .utils import StoppingCriteriaSub, download_cached_file, is_url


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


@MM_MODELS.register_module('minigpt-4-omnimmbench')
class MiniGPT4OmniMMBench(nn.Module):

    def __init__(
            self,
            vit_model='eva_clip_g',
            q_former_model='https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth',  # noqa
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision='fp16',
            freeze_vit=True,
            freeze_qformer=True,
            num_query_token=32,
            llama_model='',
            max_txt_len=32,
            end_sym='\n',
            sys_prompt='',
            low_resource=False):
        super().__init__()

        self.device = get_device()
        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint,
            vit_precision)
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = False
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = False
            logging.info('freeze vision encoder')
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for _, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = False
            self.query_tokens.requires_grad = False
            logging.info('freeze Qformer')
        print('Loading Q-Former Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model,
                                                              use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': 0})
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        for _, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(self.Qformer.config.hidden_size,
                                    self.llama_model.config.hidden_size)
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        stop_words_ids = [
            torch.tensor([835]).to(self.device),
            torch.tensor([2277, 29937]).to(self.device),
        ]
        self.stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)])
        self.sys_prompt = sys_prompt

    @classmethod
    def init_Qformer(cls,
                     num_query_token,
                     vision_width,
                     cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained('bert-base-uncased')
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0,
                                  std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(url_or_filename,
                                               check_hash=False,
                                               progress=True)
            checkpoint = torch.load(cached_file, map_location='cpu')
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location='cpu')
        else:
            raise RuntimeError('checkpoint url or path is invalid')

        state_dict = checkpoint['model']

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info('load checkpoint from %s' % url_or_filename)

        return msg

    def init_vision_encoder(cls, model_name, img_size, drop_path_rate,
                            use_grad_checkpoint, precision):
        assert model_name == 'eva_clip_g', 'vit model must be eva_clip_g for current version of MiniGPT-4'  # noqa
        visual_encoder = create_eva_vit_g(img_size, drop_path_rate,
                                          use_grad_checkpoint, precision)

        ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision

    @classmethod
    def init_tokenizer(cls):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided,
        # otherwise use torch.float16
        enable_autocast = get_device() != torch.device('cpu')

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

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
