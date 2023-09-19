import json
import os
import os.path as osp
import sys
from pathlib import Path

import clip
import mmengine
import torch
import torch.nn as nn
from mmengine.device import get_device
from timm.models.vision_transformer import Block

from opencompass.registry import MM_MODELS


def load_package():
    """Load required packages from llama_adapter_v2_multimodal7b."""
    current_file_path = os.path.abspath(__file__)
    current_folder_path = os.path.dirname(current_file_path)

    sys.path.append(os.path.join(current_folder_path, 'LLaMA-Adapter'))  # noqa
    from llama_adapter_v2_multimodal7b.llama.llama import (ModelArgs,
                                                           Transformer)
    from llama_adapter_v2_multimodal7b.llama.tokenizer import Tokenizer
    from llama_adapter_v2_multimodal7b.llama.utils import sample_top_p
    sys.path.pop(-1)

    return ModelArgs, Transformer, Tokenizer, sample_top_p


ModelArgs, Transformer, Tokenizer, sample_top_p = load_package()


class LLaMA_adapter(nn.Module):

    def __init__(self,
                 llama_ckpt_dir,
                 llama_tokenizer,
                 max_seq_len=512,
                 max_batch_size=1,
                 clip_model='ViT-L/14',
                 v_embed_dim=768,
                 v_depth=8,
                 v_num_heads=16,
                 v_mlp_ratio=4.0,
                 query_len=10,
                 query_layer=31,
                 w_bias=False,
                 w_lora=False,
                 lora_rank=16,
                 prompt_constructor=None,
                 post_processor=None):
        super().__init__()

        self.device = get_device()
        # load llama configs
        with open(os.path.join(llama_ckpt_dir, 'params.json'), 'r') as f:
            params = json.loads(f.read())
        model_args = ModelArgs(max_seq_len=max_seq_len,
                               max_batch_size=max_batch_size,
                               **params)

        # 1. clip and clip projector
        self.clip, self.clip_transform = clip.load(clip_model)

        clip_dim = self.clip.visual.proj.shape[1]
        self.clip_proj = nn.Linear(clip_dim, v_embed_dim)
        self.clip_proj_norm = nn.LayerNorm(v_embed_dim)

        self.query_len = query_len
        self.query_layer = query_layer

        # 2. visual query, blocks and projector
        self.visual_query = nn.Embedding(query_len, v_embed_dim)
        self.visual_blocks = nn.ModuleList([
            Block(v_embed_dim, v_num_heads, v_mlp_ratio, qkv_bias=True)
            for _ in range(v_depth)
        ])
        self.visual_proj = nn.Linear(v_embed_dim, model_args.dim)
        self.visual_proj_norm = nn.LayerNorm(model_args.dim)

        # 3. adapter query
        self.adapter_query = nn.Embedding(query_len * query_layer,
                                          model_args.dim)

        # 4. tokenizer
        self.tokenizer = Tokenizer(model_path=llama_tokenizer)

        # 5. llama
        model_args.vocab_size = self.tokenizer.n_words
        model_args.w_bias = w_bias
        model_args.w_lora = w_lora
        model_args.lora_rank = lora_rank
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.llama = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        ckpts = sorted(Path(llama_ckpt_dir).glob('*.pth'))
        for ckpt in ckpts:
            ckpt = torch.load(ckpt, map_location='cpu')
            self.llama.load_state_dict(ckpt, strict=False)

        self.prompt_constructor = mmengine.registry.build_from_cfg(
            prompt_constructor, MM_MODELS)
        if post_processor is not None:
            self.post_processor = mmengine.registry.build_from_cfg(
                post_processor, MM_MODELS)

    def clip_encode_image(self, x):
        # modified from CLIP
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # preserve all spatial tokens
        x = self.clip.visual.ln_post(x[:, :, :])

        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        return x

    def forward_visual(self, imgs):
        clip_feats = self.clip_encode_image(imgs)
        clip_feats = self.clip_proj_norm(self.clip_proj(clip_feats.float()))

        visual_query = self.visual_query.weight.unsqueeze(0).repeat(
            len(imgs), 1, 1)
        visual_query = torch.cat([visual_query, clip_feats], dim=1)
        for block in self.visual_blocks:
            visual_query = block(visual_query)

        visual_query = visual_query[:, :self.query_len, :]
        visual_query = self.visual_proj(visual_query)
        visual_query = self.visual_proj_norm(visual_query)

        return visual_query

    @torch.inference_mode()
    def forward(self, visual_query, tokens, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos:start_pos + seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen),
                          float('-inf'),
                          device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, start_pos, freqs_cis, mask)

        adapter = self.adapter_query.weight.reshape(self.query_layer,
                                                    self.query_len,
                                                    -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llama.layers[-1 * self.query_layer:]:
            dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
            dynamic_adapter = dynamic_adapter + visual_query
            h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
            adapter_index = adapter_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])

        return output.float()

    def pack_inputs(self, batch):
        images = [image.unsqueeze(0) for image in batch['inputs']]
        data_samples = [data_sample for data_sample in batch['data_samples']]
        images = torch.cat(images, dim=0).to(get_device())
        inputs = {'image': images, 'data_samples': data_samples}
        return inputs

    @torch.inference_mode()
    def generate(self, batch):
        max_gen_len = 256
        temperature = 0.1
        top_p = 0.75
        inputs = self.pack_inputs(batch)
        inputs = self.prompt_constructor(inputs)
        image = inputs['image']
        prompts = inputs['prompt']
        data_samples = inputs['data_samples']

        data_sample = data_samples[0]

        imgs = image

        # import pdb;pdb.set_trace()
        bsz = len(imgs)
        params = self.llama.params

        with torch.cuda.amp.autocast():
            visual_query = self.forward_visual(imgs)

        # import pdb;pdb.set_trace()
        if isinstance(prompts[0], str):
            prompts = [
                self.tokenizer.encode(x, bos=True, eos=False) for x in prompts
            ]

        # import pdb;pdb.set_trace()
        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len),
                            self.tokenizer.pad_id).cuda().long()

        # import pdb;pdb.set_trace()
        for k, t in enumerate(prompts):
            if len(t) <= total_len:
                tokens[k, :len(t)] = torch.tensor(t).cuda().long()
            else:
                tokens[k, :total_len] = torch.tensor(
                    t[:total_len]).cuda().long()

        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            with torch.cuda.amp.autocast():
                logits = self.forward(visual_query,
                                      tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(input_text_mask[:, cur_pos],
                                     tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # trick: early stop if bsz==1
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):

            # cut to max gen len
            t = t[len(prompts[i]):len(prompts[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[:t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        output_text = self.post_processor(decoded[0])
        data_sample.pred_answer = output_text
        return data_sample


@MM_MODELS.register_module('LLaMA-adapter-v2')
class LLaMA_adapter_v2(nn.Module):

    def __init__(self,
                 llama_dir,
                 prompt_constructor: dict,
                 post_processor: dict,
                 model_path: str = 'llama_adapter_v2_multimodal7b',
                 name: str = 'LORA-BIAS-7B',
                 mode: str = 'generation',
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 download_root='ckpts'):
        super().__init__()

        assert name in ['LORA-BIAS-7B', 'BIAS-7B', 'CAPTION-7B']
        # BIAS-7B or https://xxx/sha256_BIAS-7B.pth -> 7B
        llama_type = name.split('.')[0].split('-')[-1]
        llama_ckpt_dir = os.path.join(llama_dir, llama_type)
        llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')

        # load llama_adapter weights and model_cfg
        print(f'Loading LLaMA-Adapter from {llama_dir}')

        current_file_path = os.path.abspath(__file__)
        current_folder_path = os.path.dirname(current_file_path)
        model_path = osp.join(current_folder_path, 'LLaMA-Adapter', model_path)
        ckpt_root = osp.join(model_path, download_root)
        ckpt_map = {
            'LORA-BIAS-7B':
            '1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth',  # noqa: E501
            'BIAS-7B':
            '7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth',  # noqa: E501
            'CAPTION-7B':
            '5088aeb63a89746b90bcfd5cb819e1c7411b2771b267c6d131ce73e250a8abf0_CAPTION-7B.pth'  # noqa: E501
        }
        ckpt = torch.load(osp.join(ckpt_root, ckpt_map[name]),
                          map_location='cpu')

        model_cfg = ckpt.get('config', {})

        self.model = LLaMA_adapter(
            llama_ckpt_dir,
            llama_tokenzier_path,
            max_seq_len=512,
            max_batch_size=1,
            clip_model='ViT-L/14',
            v_embed_dim=768,
            v_depth=8,
            v_num_heads=16,
            v_mlp_ratio=4.0,
            query_len=10,
            query_layer=31,
            w_bias=model_cfg.get('w_bias', False),
            w_lora=model_cfg.get('w_lora', False),
            lora_rank=model_cfg.get('lora_rank', 16),
            prompt_constructor=prompt_constructor,
            post_processor=post_processor,
        )

        self.model.load_state_dict(ckpt['model'], strict=False)
        self.mode = mode

    def forward(self, batch):
        if self.mode == 'generation':
            return self.model.generate(batch)
