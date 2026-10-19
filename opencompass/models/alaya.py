from typing import Dict, List, Optional, Union

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          pipeline)

from opencompass.utils.prompt import PromptList

from .base import BaseModel, LMTemplateParser

PromptType = Union[PromptList, str]


class AlayaLM(BaseModel):
    """Model wrapper for Alaya model.

    Args:
        path (str): The name or path to Alaya model, could be a local path
            or a Huggingface model tag of Alaya.
        max_seq_len (int): The maximum length of the input sequence. Defaults
            to 2048.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.

    Note:
        Alaya has some arguments which should be fixed such as
            eos_token_id and  bad_words_ids.
        Model config should be loaded from a model config file.
        Triton is supported to accelerate the inference process.
        This class supports both Alaya Base model and Alaya Chat model.
    """

    def __init__(self,
                 path: str,
                 max_seq_len: int = 2048,
                 tokenizer_only: bool = False,
                 meta_template: Optional[Dict] = None,
                 **kwargs):

        self.template_parser = LMTemplateParser(meta_template)

        self.max_seq_len = max_seq_len
        self.tokenizer_only = tokenizer_only
        self.meta_template = meta_template

        self.name = path
        self.eos_token_id = 2
        self.bad_words_ids = 3

        self.gpu_id = '0'

        self.config = AutoConfig.from_pretrained(self.name,
                                                 trust_remote_code=True,
                                                 local_file_only=True)
        self.config.attn_config['attn_impl'] = 'triton'
        self.config.init_device = 'cuda:' + self.gpu_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            config=self.config,
            torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.name,
                                                       local_file_only=True,
                                                       padding_side='left')

        self.pipe = pipeline('text-generation',
                             model=self.model,
                             tokenizer=self.tokenizer,
                             bad_words_ids=[[self.bad_words_ids]],
                             eos_token_id=self.eos_token_id,
                             pad_token_id=self.eos_token_id,
                             device='cuda:' + self.gpu_id)

    def do_inference(self, instruction, history=[]):
        PROMPT_FORMAT = '### Instruction:\t\n{instruction}\n\n'
        OUTPUT_FORMAT = '### Output:\t\n{output} </s>'

        prompt = PROMPT_FORMAT.format(instruction=instruction)

        history2llm = []

        for i, msg in enumerate(history):
            if i % 2 == 0:  # user
                msg2llm = PROMPT_FORMAT.format(instruction=msg)
            else:  # alaya
                msg2llm = OUTPUT_FORMAT.format(output=msg)
            history2llm.append(msg2llm)

        flag = '### Output:\t\n'
        prompt2LLM = ''.join(history2llm) + prompt

        if len(prompt2LLM) >= 1500:
            prompt2LLM = prompt2LLM[-1500:]

        result = self.pipe(prompt2LLM,
                           max_new_tokens=100,
                           max_length=1900,
                           do_sample=True,
                           use_cache=True,
                           eos_token_id=self.eos_token_id,
                           pad_token_id=self.eos_token_id)

        try:
            output = result[0]['generated_text'][len(prompt2LLM):].lstrip(flag)
        except Exception:
            output = result[0]['generated_text']
        return output

    def generate(
        self,
        inputs,
        max_out_len: int = 1000,
    ) -> List[str]:
        """Generate results given a list of inputs."""
        outputs = []
        for instruction in inputs:
            output = self.do_inference(instruction=instruction)
            outputs.append(output)
        return outputs

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string."""
        return len(self.tokenizer.encode(prompt))

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Copied from .huggingface.py."""

        assert mask_length is None, 'mask_length is not supported'
        bsz = len(inputs)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        # tokenize
        prompt_tokens = [self.tokenizer.encode(x, True, False) for x in inputs]
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(params.max_seq_len, max_prompt_size)
        tokens = torch.zeros((bsz, total_len)).cuda().long()
        for k, t in enumerate(prompt_tokens):
            num_token = min(total_len, len(t))
            tokens[k, :num_token] = torch.tensor(t[-num_token:]).long()
        # forward
        outputs = self.model.forward(tokens, 0)
        # compute ppl
        shift_logits = outputs[..., :-1, :].contiguous().float()
        shift_labels = tokens[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        loss = loss_fct(shift_logits, shift_labels).view(bsz, -1)
        lens = (tokens != 0).sum(-1).cpu().numpy()
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss
