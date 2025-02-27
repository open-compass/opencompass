import os
import pdb
import copy
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from typing import Optional

from opencompass.models import HuggingFaceBaseModel
from opencompass.configs.datasets.infinitebench.infinitebench import (
    infinitebench_datasets,
)
from transformers import AutoConfig

from mmengine.device import is_npu_available


from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaAttention,
    apply_rotary_pos_emb,
)


__all__ = ["convert_kvcache_llama_heavy_recent", "LlamaAttention_heavy_hitter"]


def local_heavy_hitter_mask(attn_weights, heavy_budget):
    # attn_weights (BS, head, query, keys)
    dtype_attn_weights = attn_weights.dtype
    seq_length = attn_weights.shape[-1]
    padding_length = 0

    offset = torch.finfo(attn_weights.dtype).min
    tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        dtype_attn_weights
    )

    accumulated_attention_score = torch.sum(
        tmp_attn[:, :, padding_length : heavy_budget + padding_length, :], dim=-2
    )  # (head, keys)
    accumulated_attention_score[:, :, heavy_budget + padding_length :] = 0
    accumulated_attention_score[:, :, :padding_length] = 0

    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom[
        :,
        :,
        padding_length : heavy_budget + padding_length,
        padding_length : heavy_budget + padding_length,
    ] = True

    for token_index in range(heavy_budget + padding_length, seq_length):
        tmp_attn_index = nn.functional.softmax(
            attn_weights[:, :, token_index, :], dim=-1, dtype=torch.float32
        ).to(dtype_attn_weights)
        _, tmp_topk_index = accumulated_attention_score.topk(k=heavy_budget - 1, dim=-1)
        zeros_index = torch.zeros_like(tmp_attn_index, dtype=torch.bool)
        mask_bottom_index = zeros_index.scatter(
            -1, tmp_topk_index, True
        )  # (head, keys)
        mask_bottom_index[:, :, token_index] = True

        mask_bottom[:, :, token_index, :] = mask_bottom_index
        accumulated_attention_score += tmp_attn_index
        accumulated_attention_score = accumulated_attention_score * mask_bottom_index

    return mask_bottom


class LlamaAttention_heavy_hitter(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.rotary_emb = LlamaRotaryEmbedding(
            self.config
        )

        self.heavy_budget_ratio = config.heavy_ratio
        self.recent_budget_ratio = config.recent_ratio

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        ### Heavy + Recent
        heavy_budget = int(self.heavy_budget_ratio * attn_weights.shape[-1])
        recent_budget = int(self.recent_budget_ratio * attn_weights.shape[-1])

        # # Heavy Hitter Mask (Based on local statistics)
        # if heavy_budget > 0:
        #     mask_bottom = local_heavy_hitter_mask(attn_weights, heavy_budget) # Default: No padding applied to input
        # else:
        #     mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)

        # ones = torch.ones_like(attn_weights, dtype=torch.bool)
        # ones = torch.triu(ones, diagonal=-recent_budget)
        # mask_bottom = torch.logical_or(mask_bottom, ones)

        # mask_bottom = torch.tril(mask_bottom, diagonal=0)

        # # mask_bottom = ones
        # attn_weights[~mask_bottom] = torch.min(attention_mask)

        # Heavy Hitter Mask (Based on global statistics)
        tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            attn_weights.dtype
        )
        tmp_sum = torch.sum(tmp_attn, dim=-2)
        _, tmp_topk = tmp_sum.topk(k=heavy_budget, dim=-1)

        zeros = torch.zeros_like(tmp_sum, dtype=torch.bool)
        mask_bottom = zeros.scatter(-1, tmp_topk, True).unsqueeze(2)
        mask_bottom = mask_bottom.expand(
            mask_bottom.shape[0],
            mask_bottom.shape[1],
            attn_weights.shape[-2],
            mask_bottom.shape[-1],
        )

        ones = torch.ones_like(attn_weights, dtype=torch.bool)
        ones = torch.tril(ones, diagonal=recent_budget)
        ones = torch.triu(ones, diagonal=-recent_budget)
        mask_bottom = torch.logical_or(mask_bottom, ones)
        # mask_bottom = ones
        attn_weights[~mask_bottom] = torch.finfo(attn_weights.dtype).min

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def convert_kvcache_llama_heavy_recent(model, config):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_llama_heavy_recent(module, config)

        if isinstance(module, LlamaAttention):
            model._modules[name] = LlamaAttention_heavy_hitter(config)

    return model


ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": convert_kvcache_llama_heavy_recent,
}


def _set_model_kwargs_torch_dtype(model_kwargs):
    import torch

    if "torch_dtype" not in model_kwargs:
        torch_dtype = torch.float16
    else:
        torch_dtype = {
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.float": torch.float,
            "auto": "auto",
            "None": None,
        }.get(model_kwargs["torch_dtype"])
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    return model_kwargs


class H2OLLAMABenchmarkRunner(HuggingFaceBaseModel):
    def __init__(
        self,
        path: str,
        model_kwargs: dict = dict(),
        tokenizer_path: Optional[str] = None,
        tokenizer_kwargs: dict = dict(),
        peft_path: Optional[str] = None,
        peft_kwargs: dict = dict(),
        tokenizer_only: bool = False,
        generation_kwargs: dict = dict(),
        max_seq_len: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        stop_words: Optional[str] = [],
        drop_middle: bool = False,
        **other_kwargs,
    ):
        self.heavy_ratio = other_kwargs["heavy_ratio"]
        self.recent_ratio = other_kwargs["recent_ratio"]
        super().__init__(
            path=path,
            model_kwargs=model_kwargs,
            tokenizer_path=tokenizer_path,
            tokenizer_kwargs=tokenizer_kwargs,
            peft_path=peft_path,
            peft_kwargs=peft_kwargs,
            tokenizer_only=tokenizer_only,
            generation_kwargs=generation_kwargs,
            max_seq_len=max_seq_len,
            pad_token_id=pad_token_id,
            stop_words=stop_words,
            drop_middle=drop_middle
        )

    def _load_model(
        self,
        path: str,
        kwargs: dict,
        peft_path: Optional[str] = None,
        peft_kwargs: dict = dict(),
    ):
        # self.logger("load modified model")
        # exit(0)
        from transformers import AutoModel, AutoModelForCausalLM

        DEFAULT_MODEL_KWARGS = dict(device_map="auto", trust_remote_code=True)
        model_kwargs = DEFAULT_MODEL_KWARGS
        model_kwargs.update(kwargs)
        model_kwargs = _set_model_kwargs_torch_dtype(model_kwargs)
        self.logger.debug(f"using model_kwargs: {model_kwargs}")
        if is_npu_available():
            model_kwargs["device_map"] = "npu"

        try:
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        except ValueError:
            self.model = AutoModel.from_pretrained(path, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel

            peft_kwargs["is_trainable"] = False
            self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)

        config = AutoConfig.from_pretrained(path)
        config.heavy_ratio = self.heavy_ratio
        self.model.eval()
        self.model.generation_config.do_sample = False
