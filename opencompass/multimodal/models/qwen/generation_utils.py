# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Generation support."""

from typing import List, Tuple, Union

import torch
from transformers import PreTrainedTokenizer

# Types.
HistoryType = List[Tuple[str, str]]
TokensType = List[int]
BatchTokensType = List[List[int]]


def pad_batch(batch: BatchTokensType, pad_id: int,
              seq_length: int) -> BatchTokensType:
    for tokens in batch:
        context_length = len(tokens)
        if context_length < seq_length:
            tokens.extend([pad_id] * (seq_length - context_length))
    return batch


def get_ltor_masks_and_position_ids(
    data: torch.Tensor,
    eod_token: int,
    reset_position_ids: bool,
    reset_attention_mask: bool,
    eod_mask_loss: bool,
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(
        torch.ones((att_mask_batch, seq_length, seq_length),
                   device=data.device)).view(att_mask_batch, 1, seq_length,
                                             seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length,
                                dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modified based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indices where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indices from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indices:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= i + 1 - prev_index
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def get_batch(context_tokens: torch.LongTensor, eod_id: int):
    """Generate batch from context tokens."""
    # Move to GPU.
    tokens = context_tokens.contiguous().to(context_tokens.device)
    # Get the attention mask and position ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        eod_id,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )
    return tokens, attention_mask, position_ids


def get_stop_words_ids(chat_format: str, tokenizer: PreTrainedTokenizer):
    if chat_format == 'raw':
        stop_words_ids = [tokenizer.encode('Human:'), [tokenizer.eod_id]]
    elif chat_format == 'chatml':
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f'Unknown chat format {chat_format!r}')
    return stop_words_ids


def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = '',
    max_window_size: int = 6144,
    chat_format: str = 'chatml',
):
    if history is None:
        history = []

    if chat_format == 'chatml':
        im_start, im_end = '<|im_start|>', '<|im_end|>'
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode('\n')

        def _tokenize_str(role, content):
            return f'{role}\n{content}', tokenizer.encode(
                role, allowed_special=set(
                    tokenizer.IMAGE_ST)) + nl_tokens + tokenizer.encode(
                        content, allowed_special=set(tokenizer.IMAGE_ST))

        system_text, system_tokens_part = _tokenize_str('system', system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ''
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str('user', turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            if turn_response is not None:
                response_text, response_tokens_part = _tokenize_str(
                    'assistant', turn_response)
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens  # noqa

                next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens  # noqa
                prev_chat = (
                    f'\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}'  # noqa
                )
            else:
                next_context_tokens = nl_tokens + query_tokens + nl_tokens
                prev_chat = f'\n{im_start}{query_text}{im_end}\n'

            current_context_size = (len(system_tokens) +
                                    len(next_context_tokens) +
                                    len(context_tokens))
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f'{im_start}{system_text}{im_end}' + raw_text
        context_tokens += (nl_tokens + im_start_tokens +
                           _tokenize_str('user', query)[1] + im_end_tokens +
                           nl_tokens + im_start_tokens +
                           tokenizer.encode('assistant') + nl_tokens)
        raw_text += f'\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n'

    elif chat_format == 'raw':
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f'Unknown chat format {chat_format!r}')

    return raw_text, context_tokens


def _decode_default(
    tokens: List[int],
    *,
    stop_words: List[str],
    eod_words: List[str],
    tokenizer: PreTrainedTokenizer,
    raw_text_len: int,
    verbose: bool = False,
    return_end_reason: bool = False,
    errors: str = 'replace',
):
    trim_decode_tokens = tokenizer.decode(tokens, errors=errors)[raw_text_len:]
    if verbose:
        print('\nRaw Generate: ', trim_decode_tokens)

    end_reason = f'Gen length {len(tokens)}'
    for stop_word in stop_words:
        trim_decode_tokens = trim_decode_tokens.replace(stop_word, '').strip()
    for eod_word in eod_words:
        if eod_word in trim_decode_tokens:
            end_reason = f'Gen {eod_word!r}'
        trim_decode_tokens = trim_decode_tokens.split(eod_word)[0]
    trim_decode_tokens = trim_decode_tokens.strip()
    if verbose:
        print('\nEnd Reason:', end_reason)
        print('\nGenerate: ', trim_decode_tokens)

    if return_end_reason:
        return trim_decode_tokens, end_reason
    else:
        return trim_decode_tokens


def _decode_chatml(tokens: List[int],
                   *,
                   stop_words: List[str],
                   eod_token_ids: List[int],
                   tokenizer: PreTrainedTokenizer,
                   raw_text_len: int,
                   context_length: int,
                   verbose: bool = False,
                   return_end_reason: bool = False,
                   errors: str = 'replace'):
    end_reason = f'Gen length {len(tokens)}'
    eod_token_idx = context_length
    for eod_token_idx in range(context_length, len(tokens)):
        if tokens[eod_token_idx] in eod_token_ids:
            end_reason = f'Gen {tokenizer.decode([tokens[eod_token_idx]])!r}'
            break

    trim_decode_tokens = tokenizer.decode(tokens[:eod_token_idx],
                                          errors=errors)[raw_text_len:]
    if verbose:
        print('\nRaw Generate w/o EOD:',
              tokenizer.decode(tokens, errors=errors)[raw_text_len:])
        print('\nRaw Generate:', trim_decode_tokens)
        print('\nEnd Reason:', end_reason)
    for stop_word in stop_words:
        trim_decode_tokens = trim_decode_tokens.replace(stop_word, '').strip()
    trim_decode_tokens = trim_decode_tokens.strip()
    if verbose:
        print('\nGenerate:', trim_decode_tokens)

    if return_end_reason:
        return trim_decode_tokens, end_reason
    else:
        return trim_decode_tokens


def decode_tokens(
    tokens: Union[torch.LongTensor, TokensType],
    tokenizer: PreTrainedTokenizer,
    raw_text_len: int,
    context_length: int,
    chat_format: str,
    verbose: bool = False,
    return_end_reason: bool = False,
    errors: str = 'replace',
) -> str:
    if torch.is_tensor(tokens):
        tokens = tokens.cpu().numpy().tolist()

    if chat_format == 'chatml':
        return _decode_chatml(
            tokens,
            stop_words=[],
            eod_token_ids=[tokenizer.im_start_id, tokenizer.im_end_id],
            tokenizer=tokenizer,
            raw_text_len=raw_text_len,
            context_length=context_length,
            verbose=verbose,
            return_end_reason=return_end_reason,
            errors=errors,
        )
    elif chat_format == 'raw':
        return _decode_default(
            tokens,
            stop_words=['<|endoftext|>'],
            eod_words=['<|endoftext|>'],
            tokenizer=tokenizer,
            raw_text_len=raw_text_len,
            verbose=verbose,
            return_end_reason=return_end_reason,
            errors=errors,
        )
    else:
        raise NotImplementedError(f'Unknown chat format {chat_format!r}')
