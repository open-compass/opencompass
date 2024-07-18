from abc import abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine import dist

from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


class BaseModel:
    """Base class for model wrapper.

    Args:
        path (str): The path to the model.
        max_seq_len (int): The maximum sequence length of the model. Defaults
            to 2048.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        generation_kwargs (Dict, optional): The generation kwargs for the
            model. Defaults to dict().
        sync_rank (bool): Whether to sync inputs between ranks. Do not use this
            if you are not familiar with this behavior. Check `sync_inputs`
            function for more details. Defaults to False.
    """

    is_api: bool = False

    def __init__(self,
                 path: str,
                 max_seq_len: int = 2048,
                 tokenizer_only: bool = False,
                 meta_template: Optional[Dict] = None,
                 generation_kwargs: Optional[Dict] = dict(),
                 sync_rank: bool = False):
        self.path = path
        self.max_seq_len = max_seq_len
        self.tokenizer_only = tokenizer_only
        # meta template
        self.template_parser = LMTemplateParser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']
        self.generation_kwargs = generation_kwargs
        self.sync_rank = sync_rank

    @abstractmethod
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support'
                                  ' gen-based evaluation yet, try ppl-based '
                                  'instead.')

    @abstractmethod
    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support'
                                  ' ppl-based evaluation yet, try gen-based '
                                  'instead.')

    @abstractmethod
    def get_ppl_tokenwise(
            self,
            inputs: List[str],
            mask_length: Optional[List[int]] = None) -> List[float]:
        """Get tokenwise perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support'
                                  ' ppl-based evaluation yet, try gen-based '
                                  'instead.')

    @abstractmethod
    def encode(self, prompt: str) -> torch.Tensor:
        """Encode prompt to tokens. Not necessary for most cases.

        Args:
            prompt (str): Input string.

        Returns:
            torch.Tensor: Encoded tokens.
        """
        raise NotImplementedError(
            f'{self.__class__.__name__} does not implement'
            '`encode` method.')

    @abstractmethod
    def decode(self, tokens: torch.Tensor) -> str:
        """Decode tokens to text. Not necessary for most cases.

        Args:
            tokens (torch.Tensor): Input tokens.

        Returns:
            str: Decoded text.
        """
        raise NotImplementedError(
            f'{self.__class__.__name__} does not implement'
            '`decode` method.')

    @abstractmethod
    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """

    def parse_template(self, prompt_template: PromptType, mode: str) -> str:
        """Parse a prompt template, and wrap it with meta template if
        applicable.

        Args:
            prompt_template (List[PromptType]): A prompt
                template (potentially before being wrapped by meta template).
            mode (str): Parsing mode. Choices are 'ppl' and 'gen'.

        Returns:
            str: The final string.
        """
        return self.template_parser.parse_template(prompt_template, mode)

    def get_ppl_from_template(self,
                              templates: List[PromptType],
                              mask_length=None):
        """Get perplexity given a list of templates.

        Args:
            templates (List[PromptType]): A list of templates.
            mask_length (List[int]): A list of mask lengths. If provided, the
                perplexity will be calculated only on the unmasked tokens.
        """
        inputs = self.parse_template(templates, mode='ppl')
        return self.get_ppl(inputs, mask_length)

    def get_ppl_tokenwise_from_template(self,
                                        templates: List[PromptType],
                                        label: List[List[int]],
                                        mask_length=None):
        """Get token-wise perplexity given a list of templates.

        Args:
            templates (List[PromptType]): A list of templates.
            mask_length (List[int]): A list of mask lengths. If provided, the
                perplexity will be calculated only on the unmasked tokens.
        """
        inputs = self.parse_template(templates, mode='ppl')
        return self.get_ppl_tokenwise(inputs, label, mask_length)

    def generate_from_template(self, templates: List[PromptType],
                               max_out_len: int, **kwargs):
        """Generate completion from a list of templates.

        Args:
            templates (List[PromptType]): A list of templates.
            max_out_len (int): The maximum length of the output.
        """
        inputs = self.parse_template(templates, mode='gen')
        if hasattr(self, 'sync_rank') and self.sync_rank:
            inputs = self.sync_inputs(inputs)
        return self.generate(inputs, max_out_len=max_out_len, **kwargs)

    def get_token_len_from_template(
            self,
            templates: Union[PromptType, List[PromptType]],
            mode: str = 'ppl') -> Union[List[int], int]:
        """Get lengths given a list of templates.

        Args:
            templates (Union[List[str], str]): Input template(s).
            mode (str): Parsing mode. Choices are 'ppl' and 'gen'.

        Returns:
            Union[List[int], int]: Length(s) of the input tokens. If the input
            is a list, a list of lengths will be returned. Otherwise, an int
            will be returned.
        """
        prompts = self.parse_template(templates, mode=mode)
        assert isinstance(prompts, (list, str)), 'tokens must be list or str'
        is_batched = isinstance(prompts,
                                list) and not isinstance(prompts, PromptList)
        if not is_batched:
            prompts = [prompts]
        prompts = [str(prompt) for prompt in prompts]
        token_lens = [self.get_token_len(prompt) for prompt in prompts]
        return token_lens[0] if not is_batched else token_lens

    def sync_inputs(self, inputs: str) -> str:
        """For some case, when it involves multiprocessing with multiple gpus,
        there might be the chance that inputs are different among different
        gpus. Therefore, we need to sync inputs for rank0.

        Args:
            inputs (str): Inputs for each rank.
        """
        rank = dist.get_rank()

        if rank == 0:
            tokens = self.encode(inputs)
            length = self.get_token_len(inputs)
            if length > 2048:
                from opencompass.utils import get_logger
                get_logger().info(f'Large tokens nums: {length}')
            size = torch.tensor([tokens.shape], dtype=torch.long)
        else:
            tokens = None
            size = torch.empty(2, dtype=torch.long)

        # broadcast data size
        dist.broadcast(size, src=0)

        if rank != 0:
            tokens = torch.empty(size.tolist(), dtype=torch.long)

        # broadcast tokens
        dist.broadcast(tokens, src=0)
        # the final input might be different from original input
        # due to the max sequence limitation
        return self.decode(tokens)

    def to(self, device):
        self.model.to(device)


class LMTemplateParser:
    """Intermidate prompt template parser, specifically for language models.

    Args:
        meta_template (Dict): The meta template for the model.
    """

    def __init__(self, meta_template: Optional[Dict] = None):
        self.meta_template = meta_template
        if meta_template:
            assert 'round' in meta_template, 'round is required in meta' \
                ' template'
            assert isinstance(meta_template['round'], list)
            keys_to_check = ['round']

            if 'reserved_roles' in meta_template:
                assert isinstance(meta_template['reserved_roles'], list)
                keys_to_check.append('reserved_roles')

            self.roles: Dict[str, dict] = dict()  # maps role name to config
            for meta_key in keys_to_check:
                for item in meta_template[meta_key]:
                    assert isinstance(item, (str, dict))
                    if isinstance(item, dict):
                        assert item['role'] not in self.roles, \
                            'role in meta prompt must be unique!'
                        self.roles[item['role']] = item.copy()
                        # convert list of string and int into a raw string
                        # for the ease of future prompt processing
                        for key in ['begin', 'end']:
                            value = self.roles[item['role']].get(key, '')
                            if isinstance(value, list):
                                self.roles[item['role']][
                                    key] = self._encode_speical_tokens(value)

    def parse_template(self, prompt_template: PromptType, mode: str) -> str:
        """Parse a prompt template, and wrap it with meta template if
        applicable.

        Args:
            prompt_template (List[PromptType]): A prompt
                template (potentially before being wrapped by meta template).
            mode (str): Parsing mode. Choices are 'ppl' and 'gen'.

        Returns:
            str: The final string.
        """
        assert isinstance(prompt_template, (str, list, PromptList, tuple))
        if not isinstance(prompt_template, (str, PromptList)):
            return [self.parse_template(p, mode=mode) for p in prompt_template]

        assert mode in ['ppl', 'gen']
        if isinstance(prompt_template, str):
            return prompt_template
        if self.meta_template:

            prompt = ''
            # Whether to keep generating the prompt
            generate = True

            section_stack = []  # stores tuples: (section_name, start_idx)

            for i, item in enumerate(prompt_template):
                if not generate:
                    break
                if isinstance(item, str):
                    prompt += item
                elif isinstance(item, dict) and 'section' in item:
                    if item['pos'] == 'end':
                        section_name, start_idx = section_stack.pop(-1)
                        assert section_name == item['section']
                        if section_name in ['round', 'ice']:
                            dialogue = prompt_template[start_idx:i]
                            round_ranges = self._split_rounds(
                                dialogue, self.meta_template['round'])
                            # Consider inserting multiple round examples into
                            # template
                            for i in range(len(round_ranges) - 1):
                                start = round_ranges[i]
                                end = round_ranges[i + 1]
                                round_template = dialogue[start:end]
                                role_dict = self._update_role_dict(
                                    round_template)
                                new_str, generate = self._prompt2str(
                                    self.meta_template['round'],
                                    role_dict,
                                    # Start generating only when the mode is in
                                    # generation and the template reaches the
                                    # last round
                                    for_gen=mode == 'gen'
                                    and i == len(round_ranges) - 2
                                    and section_name == 'round')
                                prompt += new_str
                    elif item['pos'] == 'begin':
                        assert item['section'] in [
                            'begin', 'round', 'end', 'ice'
                        ]
                        section_stack.append((item['section'], i + 1))
                    else:
                        raise ValueError(f'Invalid pos {item["pos"]}')
                # if in "begin" or "end" section
                elif section_stack[-1][0] in ['begin', 'end']:
                    role_dict = self._update_role_dict(item)
                    new_str, generate = self._prompt2str(
                        item,
                        role_dict,
                        # never stop generation
                        for_gen=False)
                    prompt += new_str

            prompt = self.meta_template.get('begin', '') + prompt
            if generate:
                prompt += self.meta_template.get('end', '')

        else:
            # in case the model does not have any meta template
            prompt = ''
            last_sep = ''
            for item in prompt_template:
                if isinstance(item, dict) and set(['section', 'pos']) == set(
                        item.keys()):
                    continue
                if isinstance(item, str):
                    if item:
                        prompt += last_sep + item
                elif item.get('prompt', ''):  # it's a dict
                    prompt += last_sep + item.get('prompt', '')
                last_sep = '\n'
        return prompt

    def _split_rounds(
            self, prompt_template: List[Union[str, Dict]],
            single_round_template: List[Union[str, Dict]]) -> List[int]:
        """Split the prompt template into rounds, based on single round
        template.

        Return the index ranges of each round. Specifically,
        prompt_template[res[i]:res[i+1]] represents the i-th round in the
        template.
        """
        role_idxs = {
            role_cfg['role']: i
            for i, role_cfg in enumerate(single_round_template)
            if not isinstance(role_cfg, str)
        }
        last_role_idx = -1
        cutoff_idxs = [0]
        for idx, template in enumerate(prompt_template):
            if isinstance(template, str):
                continue
            role_idx = role_idxs[template['role']]
            if role_idx <= last_role_idx:
                cutoff_idxs.append(idx)
            last_role_idx = role_idx
        cutoff_idxs.append(len(prompt_template))
        return cutoff_idxs

    def _update_role_dict(self, prompt: Union[List, str,
                                              Dict]) -> Dict[str, Dict]:
        """Update the default role dict with the given prompt(s)."""
        assert isinstance(prompt, (str, list, dict))
        role_dict = deepcopy(self.roles)
        if isinstance(prompt, str):
            return role_dict
        if isinstance(prompt, dict):
            prompt = [prompt]
        for p in prompt:
            if isinstance(p, dict):
                role = p['role']
                if role not in self.roles:
                    role = p.get('fallback_role', None)
                    if not role:
                        print(f'{p} neither has an appropriate role nor '
                              'a fallback role.')
                role_dict[role].update(p)
        return role_dict

    def _prompt2str(self,
                    prompt: Union[List, str, Dict],
                    role_dict: Dict[str, Dict],
                    for_gen: bool = False) -> Tuple[str, bool]:
        """Convert the prompts to a string, given an updated role_dict.

        Args:
            prompts (Union[List, str, dict]): The prompt(s) to be converted.
            role_dict (Dict[str, Dict]): The updated role dict.
            for_gen (bool): If True, the prompts will be converted for
                generation tasks. The conversion stops before the first
                role whose "generate" is set to True.

        Returns:
            Tuple[str, bool]: The converted string, and whether the follow-up
            conversion should be proceeded.
        """
        assert isinstance(prompt, (list, str, dict))

        if isinstance(prompt, str):
            return prompt, True
        if isinstance(prompt, dict):
            return self._role2str(prompt, role_dict, for_gen)

        res = ''
        for p in prompt:
            new_str, cont = self._prompt2str(p, role_dict, for_gen)
            res += new_str
            if not cont:
                break
        return res, cont

    def _role2str(self,
                  role_prompt: Dict,
                  role_dict: Dict[str, Dict],
                  for_gen: bool = False) -> Tuple[str, bool]:
        """Convert a role prompt to a string, given an updated role_dict.

        Args:
            role_prompt (Dict): The role prompt to be converted.
            role_dict (Dict[str, Dict]): The updated role dict.
            for_gen (bool): If True, the prompts will be converted for
                generation tasks. The conversion stops before the first
                role whose "generate" is set to True.

        Returns:
            Tuple[str, bool]: The converted string, and whether the follow-up
            conversion should be proceeded.
        """
        merged_prompt = role_dict.get(
            role_prompt['role'],
            role_dict.get(role_prompt.get('fallback_role')))
        res = merged_prompt.get('begin', '')
        if for_gen and merged_prompt.get('generate', False):
            return res, False
        # res += merged_prompt.get('prompt', '') + merged_prompt.get('end', '')
        res += merged_prompt.get('prompt', '') + merged_prompt.get('end', '')
        return res, True

    def _encode_speical_tokens(self, prompt: List[Union[str, int]]) -> str:
        """Encode the special tokens in the prompt.

        Now this is left for the future work
        """
        raise NotImplementedError('Using List[str|int] is as the begin or end'
                                  'of a prompt is not supported yet.')
        res = ''
        for item in prompt:
            if isinstance(item, str):
                res += item
            else:
                res += f'<META_TOKEN_{item}>'
        return res
