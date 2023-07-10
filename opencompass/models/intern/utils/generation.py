r"""

"""

import torch
import torch.nn.functional as F
from torch import nn


class InferenceParams:

    def __init__(self,
                 max_sequence_len,
                 max_batch_size,
                 sequence_len_offset=0,
                 batch_size_offset=0,
                 key_value_memory_dict: dict = None,
                 lengths_per_sample=None,
                 attention_mask=None) -> None:
        """推理相关的中间 cache 对象.

        :param max_sequence_len: 最大长度
        :param max_batch_size: batch_size
        :param sequence_len_offset: _description_, defaults to 0
        :param batch_size_offset: _description_, defaults to 0
        :param key_value_memory_dict: _description_, defaults to None
        :param lengths_per_sample: _description_, defaults to None
        """
        self.max_sequence_len: int = max_sequence_len
        self.max_batch_size: int = max_batch_size
        self.sequence_len_offset: int = sequence_len_offset
        self.batch_size_offset: int = batch_size_offset
        if key_value_memory_dict is None:
            key_value_memory_dict = {}
        self.key_value_memory_dict: dict = key_value_memory_dict
        self.fused_ft_kernel: bool = False
        self.lengths_per_sample = lengths_per_sample
        self.attention_mask = attention_mask

    def reorder_state(self, indices):
        # 在 beam search 期间会会涉及到重排的操作
        if self.lengths_per_sample is not None:
            self.lengths_per_sample = self.lengths_per_sample.index_select(
                index=indices, dim=0)
        for key, value in list(self.key_value_memory_dict.items()):
            value = value.index_select(index=indices, dim=0)
            self.key_value_memory_dict[key] = value


def _get_model_device(model):
    r"""
    传入一个nn.Module的模型，获取它所在的device

    :param model: nn.Module
    :return: torch.device,None 如果返回值为None，说明这个模型没有任何参数。
    """
    assert isinstance(model, nn.Module)

    parameters = list(model.parameters())
    if len(parameters) == 0:
        return None
    else:
        return parameters[0].device


@torch.no_grad()
def _no_beam_search_generate(decoder,
                             tokens,
                             inference_params=None,
                             num_return_sequences=1,
                             max_length=20,
                             temperature=1.0,
                             top_k=50,
                             top_p=1.0,
                             eos_token_id=None,
                             do_sample=True,
                             repetition_penalty=1.0,
                             length_penalty=1.0,
                             pad_token_id=0,
                             bos_token_id=1,
                             feat_mask=None,
                             ffn_mask=None,
                             layer_mask=None):
    batch_size = tokens.size(0)
    if eos_token_id is None:
        _eos_token_id = -1
    else:
        _eos_token_id = eos_token_id

    has_bos = torch.all(tokens[:, 0].eq(bos_token_id))
    if has_bos:
        bos_pos = torch.where(tokens.eq(bos_token_id), 1, 0)
        bos_sum = bos_pos.cumsum(dim=-1)
        bos_pos = torch.where(bos_sum.eq(bos_sum[:, -1:]), 0, 1)
        to_atten_x = bos_pos[:, :, None]
        to_atten_y = bos_pos[:, None, :]

    else:
        bos_pos = torch.where(tokens.eq(bos_token_id), 1, 0)
        to_atten_x = bos_pos[:, :, None]
        to_atten_y = bos_pos[:, None, :]

    attention_mask = torch.logical_or(to_atten_x, to_atten_y).eq(1)
    if inference_params is None:
        inference_params = InferenceParams(max_sequence_len=max_length,
                                           max_batch_size=tokens.size(0),
                                           sequence_len_offset=0,
                                           batch_size_offset=0,
                                           key_value_memory_dict=None,
                                           lengths_per_sample=None,
                                           attention_mask=attention_mask)

    # 主要是为了update state

    if layer_mask is None:
        if feat_mask is None and ffn_mask is None:
            scores = decoder(**{
                'input_ids': tokens,
                'inference_params': inference_params
            })
        else:
            scores = decoder(
                **{
                    'input_ids': tokens,
                    'inference_params': inference_params,
                    'feat_mask': feat_mask,
                    'ffn_mask': ffn_mask
                })
    else:
        scores = decoder(
            **{
                'input_ids': tokens,
                'inference_params': inference_params,
                'feat_mask': feat_mask,
                'ffn_mask': ffn_mask,
                'layer_mask': layer_mask
            })

    if isinstance(scores, (list, tuple)):
        scores = scores[0]
    scores = scores[:, -1].float()
    inference_params.sequence_len_offset += tokens.size(1)
    if _eos_token_id != -1:  # 防止第一个位置为结束
        scores[:, _eos_token_id] = -1e12
    next_tokens = scores.argmax(dim=-1, keepdim=True)
    token_ids = torch.cat([tokens, next_tokens], dim=1)
    cur_len = token_ids.size(1)
    dones = token_ids.new_zeros(batch_size).eq(1)
    # tokens = tokens[:, -1:]

    real_max_length = max_length
    max_lengths = tokens.new_full((tokens.size(0), ),
                                  fill_value=max_length,
                                  dtype=torch.long)

    while cur_len < real_max_length:
        # batch_size x vocab_size
        if has_bos:
            bos_pos = torch.where(token_ids.eq(bos_token_id), 1, 0)
            bos_sum = bos_pos.cumsum(dim=-1)
            bos_pos = torch.where(bos_sum.eq(bos_sum[:, -1:]), 0, 1)
            to_atten_x = bos_pos[:, :, None]
            to_atten_y = bos_pos[:, None, :]
        else:
            bos_pos = torch.where(token_ids.eq(bos_token_id), 1, 0)
            to_atten_x = bos_pos[:, :, None]
            to_atten_y = bos_pos[:, None, :]
        # import pdb; pdb.set_trace()
        attention_mask = torch.logical_or(to_atten_x, to_atten_y).eq(1)
        # import pdb
        # pdb.set_trace()
        inference_params.attention_mask = attention_mask
        if layer_mask is None:
            if feat_mask is None and ffn_mask is None:
                scores = decoder(
                    **{
                        'input_ids': token_ids[:, -1:],
                        'inference_params': inference_params
                    })
            else:
                scores = decoder(
                    **{
                        'input_ids': token_ids[:, -1:],
                        'inference_params': inference_params,
                        'feat_mask': feat_mask,
                        'ffn_mask': ffn_mask
                    })
        else:
            scores = decoder(
                **{
                    'input_ids': token_ids[:, -1:],
                    'inference_params': inference_params,
                    'feat_mask': feat_mask,
                    'ffn_mask': ffn_mask,
                    'layer_mask': layer_mask
                })

        if isinstance(scores, (list, tuple)):
            scores = scores[0]
        scores = scores[:, -1].float()
        inference_params.sequence_len_offset += 1

        if repetition_penalty != 1.0:
            token_scores = scores.gather(dim=1, index=token_ids)
            lt_zero_mask = token_scores.lt(0).float()
            ge_zero_mask = lt_zero_mask.eq(0).float()
            token_scores = lt_zero_mask * repetition_penalty * token_scores + \
                           ge_zero_mask / repetition_penalty * token_scores  # noqa: E127 E501
            scores.scatter_(dim=1, index=token_ids, src=token_scores)

        if eos_token_id is not None and length_penalty != 1.0:
            # batch_size x vocab_size
            token_scores = scores / cur_len**length_penalty
            eos_mask = scores.new_ones(scores.size(1))
            eos_mask[eos_token_id] = 0
            eos_mask = eos_mask.unsqueeze(0).eq(1)
            # 也即除了eos，其他词的分数经过了放大/缩小
            scores = scores.masked_scatter(eos_mask, token_scores)

        if do_sample:
            if temperature > 0 and temperature != 1:
                scores = scores / temperature

            scores = top_k_top_p_filtering(scores,
                                           top_k,
                                           top_p,
                                           min_tokens_to_keep=2)
            # 加上1e-12是为了避免https://github.com/pytorch/pytorch/pull/27523
            probs = F.softmax(scores, dim=-1) + 1e-12

            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(
                1)  # batch_size
        else:
            next_tokens = torch.argmax(scores, dim=-1)  # batch_size

        # 如果已经达到对应的sequence长度了，就直接填为eos了
        if _eos_token_id != -1:
            next_tokens = next_tokens.masked_fill(max_lengths.eq(cur_len + 1),
                                                  _eos_token_id)
        next_tokens = next_tokens.masked_fill(
            dones, pad_token_id)  # 对已经搜索完成的sample做padding
        tokens = next_tokens.unsqueeze(1)

        token_ids = torch.cat([token_ids, tokens],
                              dim=-1)  # batch_size x max_len

        end_mask = next_tokens.eq(_eos_token_id)
        dones = dones.__or__(end_mask)
        cur_len += 1

        if dones.min() == 1:
            break

    return token_ids[:, None]


def top_k_top_p_filtering(logits,
                          top_k=0,
                          top_p=1.0,
                          filter_value=-float('Inf'),
                          min_tokens_to_keep=1):
    """根据top_k, top_p的值，将不满足的值置为filter_value的值。

    :param torch.Tensor logits: bsz, vocab_size
    :param int top_k: 如果大于0，则只保留最 top_k 的词汇的概率，剩下的位置被置为
        filter_value
    :param int top_p: 根据(http://arxiv.org/abs/1904.09751)设置的筛选方式
    :param float filter_value:
    :param int min_tokens_to_keep: 每个sample返回的分布中有概率的词不会低于这个值
    :return:
    """
    if top_k > 0:
        # Safety check
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        # Remove all tokens with a probability less than the last token of
        # the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1,
                                                                  None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
                                        dim=-1)

        # Remove tokens with cumulative probability above the threshold
        # (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            # (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits
