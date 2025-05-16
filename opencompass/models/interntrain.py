import os
import random
import sys
import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist

from opencompass.models.base import BaseModel
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger


class InternTrainManager:

    def __init__(self, module_path):
        self.module_path = module_path

    @staticmethod
    def build(module_path):
        sys.path.insert(0, module_path)
        try:
            from internlm.core.context.registry import \
                register_model_initializer  # noqa: F401
            return CurrentInternTrainManager(module_path)
        except ImportError:
            return LegacyInternTrainManager(module_path)


class CurrentInternTrainManager(InternTrainManager):

    def load_config(self, path, model_config=None):
        if model_config is None:
            from internlm.checkpoint.checkpoint_manager import try_load_config
            model_config = try_load_config(
                os.path.join(path, 'model_config.pt'))
        elif isinstance(model_config, str) and model_config.endswith('.pt'):
            from internlm.checkpoint.checkpoint_manager import try_load_config
            model_config = try_load_config(model_config)
        else:
            from internlm.config import Config
            if isinstance(model_config, dict):
                model_config = Config(model_config)
            elif isinstance(model_config, str):
                model_config = Config.fromfile(model_config).model
            else:
                raise NotImplementedError(
                    'model_config should be None, dict or filename.')

        return model_config

    def initialize_model(self):
        from internlm.train.pipeline import (initialize_model,
                                             initialize_parallel_communicator)
        model = initialize_model().model
        initialize_parallel_communicator(model)

        return model


class LegacyInternTrainManager(InternTrainManager):

    def load_config(self, path, model_config=None):
        from internlm.core.context import Config
        if model_config is None:
            model_config = torch.load(os.path.join(path, 'model_config.pt'))
        elif isinstance(model_config, str) and model_config.endswith('.pt'):
            model_config = torch.load(model_config)
        elif isinstance(model_config, dict):
            model_config = Config(model_config)
        elif isinstance(model_config, str):
            model_config = Config.from_file(model_config).model
        else:
            raise NotImplementedError(
                'model_config should be None, dict or filename.')

        return model_config

    def initialize_model(self):
        from internlm.train.pipeline import initialize_model
        model = initialize_model().model

        return model


@MODELS.register_module()
class InternTrain(BaseModel):
    """Model wrapper for InternTrain.

    Args:
        path (str): The name or path to HuggingFace's model.
        module_path (str): Path of InternTrain repository.
        max_seq_len (int): The maximum length of the input sequence. Defaults
            to 2048.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        tokenizer_path (str): The path to the tokenizer. Defaults to None.
        tokenizer_type: InternTrain's tokenizer type. Defaults to 'InternLM'.
        model_config (str, dict, optional): Config of model. There are several
            options for this parameter:

                - filename (str): The config items are defined in a python file
                  so the model will load configs from this file.
                - config (dict): The configuration items are defined in a dict
                  and the model will be initialized from ```model_config```.
                - None: The config is loaded from ```path```. In this case,
                  please make sure that ```path``` contains a config file named
                  ``model_config.pt``.

            Defaults to None.
        model_type: Type of model. Defaults to 'InternTrain'
        ckpt_type: The type of load function in InternTrain when checkpoints
            are loaded. Defaults to None, which means load the checkpoint
            directlywith pipeline merged.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        model_dtype: The model's dtype. If None, will use dtype defined in
            ```model_config```. Defaults to None.
        generation_kwargs (Dict, optional): The generation kwargs for the
            model. Defaults to dict().
        sync_rank (bool): Whether to sync inputs between ranks. Do not use this
            if you are not familiar with this behavior. Check `sync_inputs`
            function for more details. Defaults to False.
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'mid' represents the part of input to
            truncate. Defaults to 'none'.
        end_str (str, optional): Whether to trim generated strings with end_str
            if the model has special ending strings that are not handled well.
            Defaults to None.
    """

    def __init__(self,
                 path: str,
                 module_path: str,
                 max_seq_len: int = 2048,
                 tokenizer_only: bool = False,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_type: str = 'INTERNLM',
                 model_config: Optional[Union[str, Dict]] = None,
                 parallel_config: Optional[str] = None,
                 model_type: str = 'INTERNLM2',
                 ckpt_type: Optional[str] = None,
                 meta_template: Optional[Dict] = None,
                 model_dtype: Optional[str] = None,
                 generation_kwargs={},
                 sync_rank: bool = False,
                 mode='none',
                 end_str: Optional[str] = None):

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         tokenizer_only=tokenizer_only,
                         meta_template=meta_template,
                         sync_rank=sync_rank)

        self.logger = get_logger()
        # insert interntrain module
        self.manager = InternTrainManager.build(module_path)

        # TODO: mode is not a good name, change it both here and huggingface.py
        # mode = 'mid' is used only in longtext eval, which cut off tokens in
        # the middle
        # https://github.com/THUDM/LongBench
        assert mode in ['none', 'mid']
        self.mode = mode

        self._load_tokenizer(tokenizer_path=tokenizer_path,
                             tokenizer_type=tokenizer_type)

        if not tokenizer_only:
            self._load_model(path=path,
                             model_config=model_config,
                             parallel_config=parallel_config,
                             model_type=model_type,
                             model_dtype=model_dtype,
                             ckpt_type=ckpt_type)

        # default generation_kwargs
        assert generation_kwargs.pop('num_return_sequences', 1) == 1  # TODO
        self.generation_kwargs = {
            'temperature': 1.0,
            'top_p': 1.0,
            'top_k': 50,
            'do_sample': False,
            'repetition_penalty': 1.0,
        }
        self.generation_kwargs.update(generation_kwargs)
        self.logger.info(f'generation_kwargs: {self.generation_kwargs}')

        # generator
        from internlm.apis.inference import SequenceGenerator
        eos_token_ids = self.generation_kwargs.get('eos_token_id', [])
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        eos_token_ids.append(self.tokenizer.eos_id)
        if self.eos_token_id is not None:
            eos_token_ids.append(self.eos_token_id)
        eos_token_ids = list(set(eos_token_ids))
        self.generator = SequenceGenerator(self.model,
                                           bos_token_id=self.tokenizer.bos_id,
                                           pad_token_id=self.tokenizer.bos_id,
                                           eos_token_id=eos_token_ids)
        self.end_str = end_str

    def _load_model(self,
                    path: str,
                    model_config: Optional[str] = None,
                    parallel_config: Optional[str] = None,
                    model_type: str = 'INTERNLM2',
                    model_dtype: Optional[str] = None,
                    ckpt_type: Optional[str] = None):
        # funcs
        from internlm.checkpoint.load_funcs import (LOAD_FUNC_DICT,
                                                    merge_pp_within_tp)
        from internlm.core.context import global_context as gpc
        from internlm.initialize.launch import launch
        from internlm.utils.storage_manager import (get_storage_manager,
                                                    init_storage_manager)

        # config
        model_config = self.manager.load_config(path, model_config)
        model_config['parallel_output'] = False
        model_config['dtype'] = self._convert_dtype(model_config['dtype'],
                                                    model_dtype=model_dtype)

        world_size = int(os.getenv('WORLD_SIZE', '1'))
        tp_size = world_size  # TODO
        self.logger.info(f'world size: {world_size} tp: {tp_size}')
        if parallel_config is None:
            parallel_config = dict(zero1=dict(size=1, fsdp=False),
                                   pipeline=dict(size=1),
                                   tensor=dict(size=tp_size, mode='mtp'),
                                   sequence_parallel=False)
        config = dict(model=model_config,
                      parallel=parallel_config,
                      data=dict(use_packed_dataset=False),
                      model_type=model_type,
                      use_cuda_flash_attn=model_config.get(
                          'use_flash_attn', True))
        launch(
            config=config,
            seed=42,
            local_rank=int(os.getenv('RANK', '0')),
            rank=int(os.getenv('LOCAL_RANK', '0')),
            world_size=int(os.getenv('WORLD_SIZE', '1')),
            host=os.getenv('MASTER_ADDR', '127.0.0.1'),
            port=int(os.getenv('MASTER_PORT', random.randint(12000, 32000))),
        )
        self.logger.info(f'Config: {gpc.config}')

        self.model = self.manager.initialize_model()

        # load state dict
        try:
            get_storage_manager()
        except AssertionError:
            init_storage_manager(False, None, None)
            get_storage_manager()
        if ckpt_type is None or ckpt_type == 'internevo':
            state_dict = merge_pp_within_tp(path, del_model_prefix=True)
            load_info = self.model.load_state_dict(state_dict, strict=False)
            self.logger.info(load_info)
        else:
            load_func = LOAD_FUNC_DICT[ckpt_type]
            load_func(path, self.model)

        if 'moe' in model_type.lower():
            self.model.eval().cuda()
        else:
            self.model.to(model_config['dtype']).eval().cuda()

    def _load_tokenizer(self, tokenizer_path: str, tokenizer_type: str):
        from internlm.core.context.registry import TOKENIZER_INITIALIZER
        tokenizer_cls = TOKENIZER_INITIALIZER.get_module(tokenizer_type)
        self.tokenizer = tokenizer_cls(
            model_path=tokenizer_path,
            use_bos=True,
            use_eos=False,
        )

        # TODO use bos as pad temporarily
        if self.tokenizer.pad_id == -1:
            self.pad_id = self.tokenizer.bos_id
        else:
            self.pad_id = self.tokenizer.pad_id

    def _convert_dtype(self, default_dtype, model_dtype=None):
        if model_dtype is None:
            return default_dtype
        elif isinstance(model_dtype, torch.dtype):
            return model_dtype
        elif model_dtype == 'torch.bfloat16':
            return torch.bfloat16
        elif model_dtype in ('torch.float16', 'torch.half'):
            return torch.float16
        elif model_dtype in ('torch.float32', 'torch.float'):
            return torch.float32
        elif model_dtype in ('torch.tf32'):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            return torch.float32
        else:
            raise NotImplementedError(f'Unknown model dtype {model_dtype}')

    def get_token_len(self, prompt: str, use_bos=None, use_eos=None) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        tokens = self.tokenizer(prompt, use_bos=use_bos, use_eos=use_eos)
        return len(tokens)

    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 min_out_len: Optional[int] = None,
                 stopping_criteria: List[str] = []) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if min_out_len is None:
            # keep same with InternTrain's default value
            min_out_len = 1

        if self.mode == 'none':
            tokens = self.batch_encode(inputs,
                                       self.max_seq_len,
                                       left_padding=True)
        else:
            tokens = self.batch_encode(inputs,
                                       self.max_seq_len - max_out_len,
                                       left_padding=True)

        # random seed for pass@k
        seed = torch.tensor(time.time(), dtype=torch.int64).cuda()

        dist.broadcast(seed, src=0)
        torch.cuda.manual_seed(seed.item())
        dist.barrier()
        outputs = self.generator.generate(
            tokens,
            max_length=tokens.shape[1] + max_out_len,
            **self.generation_kwargs)  # bsz, num_return_sequences, max_length
        outputs = outputs[:, 0, tokens.shape[1]:]
        output_text = self.batch_decode(
            outputs,
            eos_token_ids=self.generator.eos_token_id,
            stopping_criteria=stopping_criteria)  # gitleaks:allow

        return output_text

    def get_ppl(self,
                input_texts: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            input_texts (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out.

        Returns:
            List[float]: A list of perplexity scores.
        """
        outputs, inputs = self.get_logits(input_texts)

        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none',
                                             ignore_index=self.pad_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (inputs != self.pad_id).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens
        return ce_loss

    def get_loglikelihood(self, input_texts: List[str],
                          conts: List[str]) -> List[float]:
        outputs, inputs = self.get_logits(input_texts)
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none',
                                             ignore_index=self.pad_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())
        lens = (inputs != self.pad_id).sum(-1).cpu().numpy()
        replaced_texts = [
            input_text.replace(cont, '')
            for input_text, cont in zip(input_texts, conts)
        ]
        replaced_lens = [
            self.get_token_len(input_text) for input_text in replaced_texts
        ]
        loglikelihoods = []
        for nloss, nlen, rlen in zip(loss, lens, replaced_lens):
            nlen, rlen = int(nlen), int(rlen)
            nloss = nloss[:nlen]
            nloss = nloss[rlen:].float().sum().cpu().detach().numpy()
            loglikelihoods.append(-nloss)
        return np.array(loglikelihoods)

    def get_mink_percent(self,
                         input_texts: List[str],
                         k: int = 20) -> List[float]:
        """https://swj0419.github.io/detect-pretrain.github.io/"""
        outputs, inputs = self.get_logits(input_texts)
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none',
                                             ignore_index=self.pad_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())
        lens = (inputs != self.pad_id).sum(-1).cpu().numpy()
        mink_percent = []
        for nloss, nlen in zip(loss, lens):
            nlen = int(nlen)
            minklen = max(nlen * k // 100, 1)
            nloss = torch.topk(loss[-nlen:], minklen, dim=-1)[0]
            nloss = -nloss.float().mean().cpu().detach().numpy()
            mink_percent.append(nloss)
        return np.array(mink_percent)

    def get_logits(self, input_texts: Union[str, List[str]]):
        tokens = self.batch_encode(input_texts, max_seq_len=self.max_seq_len)
        outputs = self.model(input_ids=tokens)
        if isinstance(outputs, tuple):
            # moe returns (hidden_states, moe_losses)
            outputs = outputs[0]
        return outputs, tokens

    def batch_encode(self,
                     input_texts: Union[str, List[str]],
                     max_seq_len: int,
                     left_padding=False):
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        tokens = [self.tokenizer(text) for text in input_texts]
        max_len = min(max_seq_len, max([len(t) for t in tokens]))
        for i in range(len(tokens)):
            cur_input = tokens[i]
            padding_len = max_len - len(cur_input)
            if self.mode == 'none':
                cur_input = cur_input[:max_len]
            elif self.mode == 'mid' and len(cur_input) > max_len:
                mid_cut_len = max_len // 2
                cur_input = cur_input[:mid_cut_len] + cur_input[-mid_cut_len:]

            if left_padding:
                # left padding with bos
                tokens[i] = [self.tokenizer.bos_id] * padding_len + cur_input
            else:
                tokens[i] = cur_input + [self.pad_id] * padding_len

        return torch.LongTensor(tokens).cuda()

    def batch_decode(self,
                     outputs,
                     eos_token_ids: List[int],
                     stopping_criteria: List[str] = []):
        # outputs: bsz, seq_len
        output_text = []
        outputs = outputs.tolist()
        for output in outputs:
            # cut off by eos_token_ids
            eos_idx = len(output)
            for eos_id in eos_token_ids:
                if eos_id in output:
                    eos_idx = min(output.index(eos_id), eos_idx)
            text = self.tokenizer.decode(output[:eos_idx])
            if self.end_str is not None:
                text = text.split(self.end_str)[0]
            for stop_word in stopping_criteria:
                text = text.split(stop_word)[0]
            output_text.append(text)

        return output_text
