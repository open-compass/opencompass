"""LLaDA (Large Language Diffusion with mAsking) model wrapper for OpenCompass.

LLaDA is a diffusion-based language model that generates text through iterative
denoising of masked tokens, rather than autoregressive next-token prediction.

Reference: https://arxiv.org/abs/2502.09992
Official repo: https://github.com/ML-GSAI/LLaDA
"""

import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


# =============================================================================
# Inlined generation functions from LLaDA's generate.py
# =============================================================================

def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply Gumbel noise for sampling from categorical distributions.

    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves
    perplexity score but reduces generation quality. Thus, we use float64.

    Args:
        logits: Model output logits.
        temperature: Sampling temperature. If 0, returns logits unchanged.

    Returns:
        Logits with Gumbel noise applied.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _get_num_transfer_tokens(
    mask_index: torch.Tensor, steps: int
) -> torch.Tensor:
    """Compute the number of tokens to transition at each diffusion step.

    In the reverse process, the interval [0, 1] is uniformly discretized into
    steps intervals. Because LLaDA employs a linear noise schedule (Eq. 8),
    the expected number of tokens transitioned at each step should be consistent.

    Args:
        mask_index: Boolean tensor indicating masked positions.
        steps: Number of diffusion steps.

    Returns:
        Tensor of shape (batch_size, steps) with token counts per step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def _llada_generate(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = 'low_confidence',
    mask_id: int = 126336,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
) -> torch.Tensor:
    """Generate text using LLaDA's diffusion-based decoding.

    Args:
        model: The LLaDA model.
        prompt: Input tensor of shape (batch_size, prompt_length).
        attention_mask: Optional attention mask.
        steps: Number of diffusion sampling steps.
        gen_length: Length of text to generate.
        block_length: Block length for semi-autoregressive generation.
            If less than gen_length, enables semi-autoregressive remasking.
        temperature: Sampling temperature for Gumbel noise.
        cfg_scale: Classifier-free guidance scale. 0 means no guidance.
        remasking: Remasking strategy - 'low_confidence' or 'random'.
        mask_id: Token ID for [MASK] token (default 126336 for LLaDA).
        logits_eos_inf: Whether to set EOS token logits to -inf.
        confidence_eos_eot_inf: Whether to set EOS/EOT confidence to -inf.

    Returns:
        Generated token tensor of shape (batch_size, prompt_length + gen_length).
    """
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device

    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (prompt.shape[0], gen_length),
                    dtype=attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=-1,
        )

    prompt_index = x != mask_id

    assert gen_length % block_length == 0, \
        f'gen_length ({gen_length}) must be divisible by block_length ({block_length})'
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0, \
        f'steps ({steps}) must be divisible by num_blocks ({num_blocks})'
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = _get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                else:
                    attention_mask_ = None
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = -torch.inf
                logits_with_noise[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(f'Unknown remasking strategy: {remasking}')

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=int(num_transfer_tokens[j, i].item()))
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


# =============================================================================
# Helper functions
# =============================================================================

def _get_meta_template(meta_template: Optional[Dict]) -> APITemplateParser:
    """Get the meta template parser with default chat template."""
    default_meta_template = dict(
        round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ],
        reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
    )
    return APITemplateParser(meta_template or default_meta_template)


def _convert_chat_messages(
    inputs: List[Union[str, PromptList]],
    merge_role: bool = True,
    skip_empty_prompt: bool = True,
) -> List[List[Dict[str, str]]]:
    """Convert inputs to chat message format for instruct models.

    Args:
        inputs: List of string prompts or PromptList objects.
        merge_role: Whether to merge consecutive messages with same role.
        skip_empty_prompt: Whether to skip empty prompts.

    Returns:
        List of message lists, each containing role/content dicts.
    """
    outputs = []
    for _input in inputs:
        messages = []
        if isinstance(_input, str):
            messages.append({'role': 'user', 'content': _input})
        else:
            for item in _input:
                if skip_empty_prompt and not item.get('prompt'):
                    continue
                role = {
                    'HUMAN': 'user',
                    'BOT': 'assistant',
                    'SYSTEM': 'system',
                }.get(item['role'].upper(), 'user')
                messages.append({'role': role, 'content': item['prompt']})

        if merge_role:
            merged_messages = []
            for item in messages:
                if merged_messages and merged_messages[-1]['role'] == item['role']:
                    merged_messages[-1]['content'] += '\n' + item['content']
                else:
                    merged_messages.append(item)
            messages = merged_messages

        outputs.append(messages)
    return outputs


def _convert_base_messages(inputs: List[Union[str, PromptList]]) -> List[str]:
    """Convert inputs to plain text format for base models.

    Args:
        inputs: List of string prompts or PromptList objects.

    Returns:
        List of concatenated prompt strings.
    """
    outputs = []
    for _input in inputs:
        if isinstance(_input, str):
            outputs.append(_input)
        else:
            messages = []
            for item in _input:
                messages.append(item.get('prompt', ''))
            outputs.append(''.join(messages))
    return outputs


# =============================================================================
# Model classes
# =============================================================================

@MODELS.register_module()
class LLaDAModel(BaseModel):
    """Model wrapper for LLaDA instruct models.

    This wrapper supports LLaDA's diffusion-based text generation for
    instruction-tuned models that use chat templates.

    Args:
        path: Path or HuggingFace model ID for the LLaDA model.
        hf_cache_dir: HuggingFace cache directory. Defaults to HF_MODEL_HUB env.
        max_seq_len: Maximum sequence length. Defaults to 2048.
        tokenizer_path: Path to tokenizer. Defaults to model path.
        tokenizer_kwargs: Additional tokenizer arguments.
        peft_path: Path to PEFT adapter weights.
        tokenizer_only: If True, only load tokenizer.
        model_kwargs: Arguments passed to model loader.
        generation_kwargs: Additional generation arguments.
        meta_template: Meta template for prompt formatting.
        extract_pred_after_decode: Extract prediction after decoding.
        batch_padding: Enable batch padding for inference.
        pad_token_id: Override pad token ID.
        mode: Truncation mode - 'none' or 'mid'.
        use_fastchat_template: Use fastchat conversation template.
        end_str: String to trim from generated output.
        stop_words: List of stop words to truncate output.
        cfg: Classifier-free guidance scale.
        temperature: Sampling temperature.
        remasking: Remasking strategy - 'low_confidence' or 'random'.
        mask_id: Token ID for [MASK] (default 126336).
        padding_id: Token ID for <pad> (default 126081).
        mc_num: Monte Carlo samples (unused, for compatibility).
        gen_steps: Number of diffusion steps.
        gen_length: Generation length.
        gen_blocksize: Block size for semi-autoregressive generation.
        batch_size_: Internal batch size parameter.
        diff_confidence_eos_eot_inf: Set EOS/EOT confidence to -inf.
        diff_logits_eos_inf: Set EOS logits to -inf.
    """

    def __init__(
        self,
        path: str,
        hf_cache_dir: Optional[str] = None,
        max_seq_len: int = 2048,
        tokenizer_path: Optional[str] = None,
        tokenizer_kwargs: dict = dict(),
        peft_path: Optional[str] = None,
        tokenizer_only: bool = False,
        model_kwargs: dict = dict(device_map='auto'),
        generation_kwargs: dict = dict(),
        meta_template: Optional[Dict] = None,
        extract_pred_after_decode: bool = False,
        batch_padding: bool = False,
        pad_token_id: Optional[int] = None,
        mode: str = 'none',
        use_fastchat_template: bool = False,
        end_str: Optional[str] = None,
        stop_words: Optional[List[str]] = None,
        # LLaDA-specific parameters
        cfg: float = 0.0,
        temperature: float = 0.0,
        remasking: str = 'low_confidence',
        mask_id: int = 126336,
        padding_id: int = 126081,
        mc_num: int = 1,
        gen_steps: int = 512,
        gen_length: int = 512,
        gen_blocksize: int = 512,
        batch_size_: int = 1,
        diff_confidence_eos_eot_inf: bool = False,
        diff_logits_eos_inf: bool = False,
    ) -> None:
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            tokenizer_only=tokenizer_only,
            meta_template=meta_template,
        )

        if hf_cache_dir is None:
            hf_cache_dir = os.getenv('HF_MODEL_HUB', None)

        self.logger = get_logger()
        self.pad_token_id = pad_token_id
        self.hf_cache_dir = hf_cache_dir

        assert mode in ['none', 'mid'], f'mode must be "none" or "mid", got {mode}'
        self.mode = mode

        self._load_tokenizer(
            path=path,
            tokenizer_path=tokenizer_path,
            tokenizer_kwargs=tokenizer_kwargs,
        )

        self.batch_padding = batch_padding
        self.extract_pred_after_decode = extract_pred_after_decode

        if not tokenizer_only:
            self._load_model(
                path=path,
                model_kwargs=model_kwargs,
                peft_path=peft_path,
            )

        self.generation_kwargs = generation_kwargs
        self.use_fastchat_template = use_fastchat_template
        self.end_str = end_str
        self.stop_words = stop_words or []

        # LLaDA-specific parameters
        self.cfg = cfg
        self.mc_num = mc_num
        self.batch_size_ = batch_size_
        self.gen_steps = gen_steps
        self.gen_length = gen_length
        self.gen_blocksize = gen_blocksize
        self.temperature = temperature
        self.remasking = remasking
        self.padding_id = padding_id
        self.mask_id = mask_id
        self.diff_confidence_eos_eot_inf = diff_confidence_eos_eot_inf
        self.diff_logits_eos_inf = diff_logits_eos_inf

        self.template_parser = _get_meta_template(meta_template)

    def _load_tokenizer(
        self,
        path: str,
        tokenizer_path: Optional[str],
        tokenizer_kwargs: dict,
    ) -> None:
        """Load and configure the tokenizer."""
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path else path,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

        # Handle pad_token_id configuration
        if self.pad_token_id is not None:
            if self.pad_token_id < 0:
                self.pad_token_id += self.tokenizer.vocab_size
            if self.tokenizer.pad_token_id is None:
                self.logger.debug(f'Using {self.pad_token_id} as pad_token_id')
            elif self.tokenizer.pad_token_id != self.pad_token_id:
                self.logger.warning(
                    f'pad_token_id inconsistent with tokenizer. '
                    f'Using {self.pad_token_id} as pad_token_id'
                )
            self.tokenizer.pad_token_id = self.pad_token_id
        elif self.tokenizer.pad_token_id is None:
            self.logger.warning('pad_token_id is not set for the tokenizer.')
            if self.tokenizer.eos_token is not None:
                self.logger.warning(
                    f'Using eos_token_id {self.tokenizer.eos_token} as pad_token_id.'
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                from transformers.generation import GenerationConfig

                gcfg = GenerationConfig.from_pretrained(path)
                if gcfg.pad_token_id is not None:
                    self.logger.warning(
                        f'Using pad_token_id {gcfg.pad_token_id} from GenerationConfig.'
                    )
                    self.tokenizer.pad_token_id = gcfg.pad_token_id
                else:
                    raise ValueError(
                        'pad_token_id is not set for this tokenizer. '
                        'Set it via `pad_token_id={ID}` in model config.'
                    )

    def _set_model_kwargs_torch_dtype(self, model_kwargs: dict) -> None:
        """Configure torch_dtype in model_kwargs."""
        if 'torch_dtype' not in model_kwargs:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = {
                'torch.float16': torch.float16,
                'torch.bfloat16': torch.bfloat16,
                'torch.float': torch.float,
                'auto': 'auto',
                'None': None,
            }.get(model_kwargs['torch_dtype'])
        self.logger.debug(f'Using torch_dtype: {torch_dtype}')
        if torch_dtype is not None:
            model_kwargs['torch_dtype'] = torch_dtype

    def _load_model(
        self,
        path: str,
        model_kwargs: dict,
        peft_path: Optional[str] = None,
    ) -> None:
        """Load the LLaDA model."""
        from transformers import AutoModel, AutoModelForCausalLM

        self._set_model_kwargs_torch_dtype(model_kwargs)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                path, trust_remote_code=True, **model_kwargs
            )
        except ValueError:
            self.model = AutoModel.from_pretrained(
                path, trust_remote_code=True, **model_kwargs
            )

        if peft_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model, peft_path, is_trainable=False
            )

        self.model.eval()

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate text responses for a batch of inputs.

        Args:
            inputs: List of input prompts.
            max_out_len: Maximum output length (overridden by gen_length).

        Returns:
            List of generated response strings.
        """
        messages = _convert_chat_messages(inputs)
        prompts = [
            self.tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False
            )
            for m in messages
        ]

        self.logger.debug(
            f'LLaDA generate: steps={self.gen_steps}, length={self.gen_length}, '
            f'blocksize={self.gen_blocksize}, temperature={self.temperature}, '
            f'cfg={self.cfg}, remasking={self.remasking}'
        )

        self.tokenizer.padding_side = 'left'
        encoded = self.tokenizer.batch_encode_plus(
            prompts, padding=True, return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.model.device)
        attention_mask = encoded.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        output_ids = _llada_generate(
            model=self.model,
            prompt=input_ids,
            attention_mask=attention_mask,
            steps=self.gen_steps,
            gen_length=self.gen_length,
            block_length=self.gen_blocksize,
            temperature=self.temperature,
            cfg_scale=self.cfg,
            remasking=self.remasking,
            mask_id=self.mask_id,
            confidence_eos_eot_inf=self.diff_confidence_eos_eot_inf,
            logits_eos_inf=self.diff_logits_eos_inf,
        )

        responses = []
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            response = self.tokenizer.decode(
                output_ids[i, -self.gen_length :], skip_special_tokens=True
            )
            responses.append(response)

        self.logger.debug(f'Generated {len(responses)} responses')

        return responses

    def get_ppl(
        self,
        inputs: List[str],
        mask_length: Optional[List[int]] = None,
    ) -> List[float]:
        """Get perplexity scores.

        Note: LLaDA uses diffusion-based generation, not autoregressive.
        Standard perplexity calculation doesn't apply directly.
        Use lm-eval toolkit for perplexity-based evaluation.

        Raises:
            NotImplementedError: Always raised for LLaDA models.
        """
        raise NotImplementedError(
            'LLaDA is a diffusion model. Standard perplexity calculation '
            'does not apply. Use lm-eval toolkit for PPL-based evaluation, '
            'or use generation-based evaluation (e.g., gsm8k_gen).'
        )

    def get_token_len(self, prompt: str) -> int:
        """Get the token length of a prompt.

        Args:
            prompt: Input string.

        Returns:
            Number of tokens in the prompt.
        """
        return len(self.tokenizer.encode(prompt))


@MODELS.register_module()
class LLaDABaseModel(LLaDAModel):
    """Model wrapper for LLaDA base models (without chat template).

    This variant is for base models that don't use instruction formatting.
    It processes prompts as plain text without chat template wrapping.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.template_parser = LMTemplateParser(None)

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate text responses for a batch of inputs.

        Args:
            inputs: List of input prompts (plain text, no chat formatting).
            max_out_len: Maximum output length (overridden by gen_length).

        Returns:
            List of generated response strings.
        """
        prompts = _convert_base_messages(inputs)

        self.logger.debug(
            f'LLaDA base generate: steps={self.gen_steps}, length={self.gen_length}, '
            f'blocksize={self.gen_blocksize}, temperature={self.temperature}, '
            f'cfg={self.cfg}, remasking={self.remasking}'
        )

        self.tokenizer.padding_side = 'left'
        encoded = self.tokenizer.batch_encode_plus(
            prompts, padding=True, return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.model.device)
        attention_mask = encoded.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        output_ids = _llada_generate(
            model=self.model,
            prompt=input_ids,
            attention_mask=attention_mask,
            steps=self.gen_steps,
            gen_length=self.gen_length,
            block_length=self.gen_blocksize,
            temperature=self.temperature,
            cfg_scale=self.cfg,
            remasking=self.remasking,
            mask_id=self.mask_id,
            confidence_eos_eot_inf=self.diff_confidence_eos_eot_inf,
            logits_eos_inf=self.diff_logits_eos_inf,
        )

        responses = []
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            response = self.tokenizer.decode(
                output_ids[i, -self.gen_length :], skip_special_tokens=True
            )

            # Apply stop words
            for stop_word in self.stop_words:
                if stop_word in response:
                    response = response.split(stop_word)[0]
                    break

            responses.append(response)

        self.logger.debug(f'Generated {len(responses)} responses')

        return responses

    def get_token_len(self, prompt: str, add_special_tokens: bool = True) -> int:
        """Get the token length of a prompt.

        Args:
            prompt: Input string.
            add_special_tokens: Whether to add special tokens.

        Returns:
            Number of tokens in the prompt.
        """
        text = _convert_base_messages([prompt])[0]
        tokens = self.tokenizer(text, add_special_tokens=add_special_tokens)
        return len(tokens['input_ids'])
