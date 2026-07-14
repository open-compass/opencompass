# flake8: noqa
# yapf: disable
from typing import Dict, List, Optional

import numpy as np

from opencompass.models.base import BaseModel
from opencompass.utils import get_logger

from .huggingface_above_v4_33 import (_convert_chat_messages,
                                      _format_with_fast_chat_template,
                                      _get_meta_template,
                                      _get_possible_max_seq_len)

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM, SamplingParams = None, None


class VLLMwithChatTemplate(BaseModel):
    """vLLM model wrapper with chat template support.

    This class extends the base vLLM wrapper to automatically apply chat templates
    using tokenizer.apply_chat_template(), and supports LoRA adapters.
    """

    def __init__(
        self,
        path: str,
        model_kwargs: dict = dict(),
        tokenizer_only: bool = False,
        generation_kwargs: dict = dict(),
        max_seq_len: int = None,
        meta_template: Optional[Dict] = None,
        fastchat_template: Optional[str] = None,
        stop_words: List[str] = [],
        lora_path: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
    ):
        """Initialize the VLLMwithChatTemplate model.

        Args:
            path (str): Path to the base model.
            model_kwargs (dict): Additional kwargs for vLLM model initialization.
            tokenizer_only (bool): Whether to only load the tokenizer.
            generation_kwargs (dict): Default generation parameters.
            max_seq_len (int): Maximum sequence length.
            meta_template (Dict): Meta template for prompt formatting.
            fastchat_template (str): Optional fastchat template name.
            stop_words (List[str]): Additional stop words for generation.
            lora_path (str): Path to LoRA adapter weights. If provided, the model
                will use the LoRA adapter during generation.
            chat_template_kwargs (dict): Additional kwargs to pass to
                tokenizer.apply_chat_template(). For example, for Qwen3 models,
                you can pass {'enable_thinking': True/False} to control the
                thinking mode.
        """
        assert LLM, ('Please install VLLM with `pip install vllm`. note: torch==2.1.2 is required.')

        self.logger = get_logger()
        self.path = path
        self.tokenizer_only = tokenizer_only
        self.template_parser = _get_meta_template(meta_template)
        self.max_seq_len = _get_possible_max_seq_len(max_seq_len, path)
        if tokenizer_only:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        else:
            self._load_model(path, model_kwargs)
            self.tokenizer = self.model.get_tokenizer()

        self.generation_kwargs = generation_kwargs
        self.generation_kwargs.pop('do_sample', None)
        self.fastchat_template = fastchat_template
        self.stop_words = list(set(stop_words + self._get_potential_stop_words(path)))
        self.lora_path = lora_path
        self.chat_template_kwargs = chat_template_kwargs or {}

    def _load_model(self, path: str, added_model_kwargs: dict = dict()):
        import ray

        if ray.is_initialized():
            self.logger.info('shutdown ray instance to avoid "Calling ray.init() again" error.')
            ray.shutdown()

        DEFAULT_MODEL_KWARGS = dict(trust_remote_code=True)
        model_kwargs = DEFAULT_MODEL_KWARGS.copy()
        model_kwargs.update(added_model_kwargs)
        self.model = LLM(path, **model_kwargs)

    def _get_potential_stop_words(self, path: Optional[str]):
        from transformers import GenerationConfig
        potential_stop_words = []
        try:
            generation_config = GenerationConfig.from_pretrained(path)
        except:
            generation_config = None
        if generation_config and hasattr(generation_config, 'eos_token_id'):
            if isinstance(generation_config.eos_token_id, int):
                potential_stop_words.append(self.tokenizer.decode(generation_config.eos_token_id))
            else:
                assert isinstance(generation_config.eos_token_id, list)
                for token_id in generation_config.eos_token_id:
                    potential_stop_words.append(self.tokenizer.decode(token_id))
        if self.tokenizer.eos_token is not None:
            potential_stop_words.append(self.tokenizer.eos_token)
        potential_stop_words = list(set(potential_stop_words))
        potential_stop_words = [s for s in potential_stop_words if s]
        return potential_stop_words

    def generate(self, inputs: List[str], max_out_len: int, stopping_criteria: List[str] = [], **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        messages = _convert_chat_messages(inputs)
        if self.fastchat_template:
            messages = _format_with_fast_chat_template(messages, self.fastchat_template)
        else:
            messages = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False, **self.chat_template_kwargs) for m in messages]
            # vLLM tokenize prompts by AutoTokenizer with its default parameter "add_special_token=True"
            # OC add bos_token in the prompt, which requires tokenizing prompts using "add_speicial_token=False"
            # But vLLM doesn't have "add_speicial_token" in the pipeline API. So, we remove bos_token
            # from messages as a workaround
            if self.tokenizer.bos_token:
                bos_token = self.tokenizer.bos_token
                messages = [message.removeprefix(bos_token) if message.startswith(bos_token) else message for message in messages]
        DEFAULT_GENERATION_KWARGS = {
            'temperature': 0,
            'max_tokens': max_out_len,
            'stop': list(set(self.stop_words + stopping_criteria))
        }
        sampling_kwargs = DEFAULT_GENERATION_KWARGS.copy()
        sampling_kwargs.update(self.generation_kwargs)
        sampling_kwargs.update(kwargs)
        sampling_kwargs = SamplingParams(**sampling_kwargs)
        self.logger.info('Sampling Params of vLLM: ')
        self.logger.info(sampling_kwargs)

        if self.lora_path:
            try:
                from vllm.lora.request import LoRARequest
            except ImportError:
                raise ImportError('Please install vLLM with LoRA support to use lora_path parameter.')
            outputs = self.model.generate(messages, sampling_kwargs, lora_request=LoRARequest('lora_adapter', 1, self.lora_path))
        else:
            outputs = self.model.generate(messages, sampling_kwargs)

        prompt_list, output_strs = [], []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            prompt_list.append(prompt)
            output_strs.append(generated_text)

        return output_strs

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        m = _convert_chat_messages([prompt])[0]
        t = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_dict=True)
        return len(t['input_ids'])
