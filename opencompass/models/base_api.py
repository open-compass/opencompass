import re
import sys
import threading
import time
import warnings
from abc import abstractmethod
from copy import deepcopy
from queue import Queue
from time import sleep
from typing import Dict, List, Optional, Tuple, Union

from opencompass.utils import get_logger
from opencompass.utils.prompt import PromptList

from .base import BaseModel

PromptType = Union[PromptList, str]


class BaseAPIModel(BaseModel):
    """Base class for API model wrapper.

    Args:
        path (str): The path to the model.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        max_seq_len (int): The maximum sequence length of the model. Defaults
            to 2048.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        generation_kwargs (Dict, optional): The generation kwargs for the
            model. Defaults to dict().
    """

    is_api: bool = True

    def __init__(self,
                 path: str,
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 generation_kwargs: Dict = dict()):
        self.path = path
        self.max_seq_len = max_seq_len
        self.meta_template = meta_template
        self.retry = retry
        self.query_per_second = query_per_second
        self.token_bucket = TokenBucket(query_per_second, rpm_verbose)
        self.template_parser = APITemplateParser(meta_template)
        self.logger = get_logger()
        self.generation_kwargs = generation_kwargs

    @abstractmethod
    def generate(self, inputs: List[PromptType],
                 max_out_len: int) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support'
                                  ' gen-based evaluation yet, try ppl-based '
                                  'instead.')

    def flush(self):
        """Ensure simultaneous emptying of stdout and stderr when concurrent
        resources are available.

        When employing multiprocessing with standard I/O redirected to files,
        it is crucial to clear internal data for examination or prevent log
        loss in case of system failures."
        """
        if hasattr(self, 'tokens'):
            sys.stdout.flush()
            sys.stderr.flush()

    def acquire(self):
        """Acquire concurrent resources if exists.

        This behavior will fall back to wait with query_per_second if there are
        no concurrent resources.
        """
        if hasattr(self, 'tokens'):
            self.tokens.acquire()
        else:
            self.wait()

    def release(self):
        """Release concurrent resources if acquired.

        This behavior will fall back to do nothing if there are no concurrent
        resources.
        """
        if hasattr(self, 'tokens'):
            self.tokens.release()

    @abstractmethod
    def get_ppl(self,
                inputs: List[PromptType],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings.
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

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string. Only English and Chinese
        characters are counted for now. Users are encouraged to override this
        method if more accurate length is needed.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """

        english_parts = re.findall(r'[A-Za-z0-9]+', prompt)
        chinese_parts = re.findall(r'[\u4e00-\u9FFF]+', prompt)

        # Count English words
        english_count = sum(len(part.split()) for part in english_parts)

        # Count Chinese words
        chinese_count = sum(len(part) for part in chinese_parts)

        return english_count + chinese_count

    def wait(self):
        """Wait till the next query can be sent.

        Applicable in both single-thread and multi-thread environments.
        """
        return self.token_bucket.get_token()

    def to(self, device):
        pass


class APITemplateParser:
    """Intermidate prompt template parser, specifically for API models.

    Args:
        meta_template (Dict): The meta template for the model.
    """

    def __init__(self, meta_template: Optional[Dict] = None):
        self.meta_template = meta_template
        # Check meta template
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

    def parse_template(self, prompt_template: PromptType,
                       mode: str) -> PromptType:
        """Parse the intermidate prompt template, and wrap it with meta
        template if applicable. When the meta template is set and the input is
        a PromptList, the return value will be a PromptList containing the full
        conversation history. Each item looks like:

        .. code-block:: python

            {'role': 'user', 'prompt': '...'}).

        Args:
            prompt_template (List[PromptType]): An intermidate prompt
                template (potentially before being wrapped by meta template).
            mode (str): Parsing mode. Choices are 'ppl' and 'gen'.

        Returns:
            List[PromptType]: The finalized prompt or a conversation.
        """
        assert isinstance(prompt_template, (str, list, PromptList, tuple))

        if not isinstance(prompt_template, (str, PromptList)):
            return [self.parse_template(p, mode=mode) for p in prompt_template]

        assert mode in ['ppl', 'gen']
        if isinstance(prompt_template, str):
            return prompt_template
        if self.meta_template:

            prompt = PromptList()
            # Whether to keep generating the prompt
            generate = True

            section_stack = []  # stores tuples: (section_name, start_idx)

            for i, item in enumerate(prompt_template):
                if not generate:
                    break
                if isinstance(item, str):
                    if item.strip():
                        # TODO: logger
                        warnings.warn('Non-empty string in prompt template '
                                      'will be ignored in API models.')
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
                                api_prompts, generate = self._prompt2api(
                                    self.meta_template['round'],
                                    role_dict,
                                    # Start generating only when the mode is in
                                    # generation and the template reaches the
                                    # last round
                                    for_gen=mode == 'gen'
                                    and section_name == 'round'
                                    and i == len(round_ranges) - 2)
                                prompt += api_prompts
                    elif item['pos'] == 'begin':
                        assert item['section'] in [
                            'begin', 'round', 'end', 'ice'
                        ]
                        section_stack.append((item['section'], i + 1))
                    else:
                        raise ValueError(f'Invalid pos {item["pos"]}')
                elif section_stack[-1][0] in ['begin', 'end']:
                    role_dict = self._update_role_dict(item)
                    api_prompts, generate = self._prompt2api(
                        item, role_dict, for_gen=mode == 'gen')
                    prompt.append(api_prompts)

            # merge the consecutive prompts assigned to the same role
            new_prompt = PromptList([prompt[0]])
            last_role = prompt[0]['role']
            for item in prompt[1:]:
                if item['role'] == last_role:
                    new_prompt[-1]['prompt'] += '\n' + item['prompt']
                else:
                    last_role = item['role']
                    new_prompt.append(item)
            prompt = new_prompt

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
                elif item.get('prompt', ''):
                    prompt += last_sep + item.get('prompt', '')
                last_sep = '\n'
        return prompt

    def _update_role_dict(self, prompts: Union[List, str]) -> Dict[str, Dict]:
        """Update the default role dict with the given prompts."""
        role_dict = deepcopy(self.roles)
        if isinstance(prompts, str):
            return role_dict
        elif isinstance(prompts, dict):
            prompts = [prompts]
        for prompt in prompts:
            if isinstance(prompt, dict):
                role = prompt['role']
                if role not in self.roles:
                    role = prompt.get('fallback_role', None)
                    if not role:
                        print(f'{prompt} neither has an appropriate role nor '
                              'a fallback role.')
                role_dict[role].update(prompt)
        return role_dict

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
            role_idx = role_idxs.get(template['role'], None)
            if role_idx is None:
                try:
                    role_idx = role_idxs[template['fallback_role']]
                except KeyError:
                    raise KeyError(f'{template} neither has an appropriate '
                                   'role nor a fallback role.')
            if role_idx <= last_role_idx:
                cutoff_idxs.append(idx)
            last_role_idx = role_idx
        cutoff_idxs.append(len(prompt_template))
        return cutoff_idxs

    def _prompt2api(self,
                    prompts: Union[List, str],
                    role_dict: Dict[str, Dict],
                    for_gen: bool = False) -> Tuple[List, bool]:
        """Convert the prompts to a API-style prompts, given an updated
        role_dict.

        Args:
            prompts (Union[List, str]): The prompts to be converted.
            role_dict (Dict[str, Dict]): The updated role dict.
            for_gen (bool): If True, the prompts will be converted for
                generation tasks. The conversion stops before the first
                role whose "generate" is set to True.

        Returns:
            Tuple[List, bool]: The converted string, and whether the follow-up
            conversion should be proceeded.
        """
        cont = True
        if isinstance(prompts, str):
            return prompts, cont
        elif isinstance(prompts, dict):
            api_role, cont = self._role2api_role(prompts, role_dict, for_gen)
            return api_role, cont

        res = []
        for prompt in prompts:
            if isinstance(prompt, str):
                raise TypeError('Mixing str without explicit role is not '
                                'allowed in API models!')
            else:
                api_role, cont = self._role2api_role(prompt, role_dict,
                                                     for_gen)
                if api_role:
                    res.append(api_role)
                if not cont:
                    break
        return res, cont

    def _role2api_role(self,
                       role_prompt: Dict,
                       role_dict: Dict[str, Dict],
                       for_gen: bool = False) -> Tuple[Dict, bool]:
        """Convert a role prompt to a string, given an updated role_dict.

        Args:
            role_prompt (Dict): The role prompt to be converted.
            role_dict (Dict[str, Dict]): The updated role dict.
            for_gen (bool): If True, the prompts will be converted for
                generation tasks. The conversion stops before the first
                role whose "generate" is set to True.

        Returns:
            Tuple[Dict, bool]: The converted string, and whether the follow-up
            conversion should be proceeded.
        """
        merged_prompt = role_dict.get(
            role_prompt['role'],
            role_dict.get(role_prompt.get('fallback_role')))
        # res_api_prompt = dict(type='', )
        if for_gen and merged_prompt.get('generate', False):
            return None, False
        res = {}
        res['role'] = merged_prompt['api_role']
        res['prompt'] = merged_prompt.get('begin', '')
        res['prompt'] += merged_prompt.get('prompt', '')
        res['prompt'] += merged_prompt.get('end', '')
        return res, True


class TokenBucket:
    """A token bucket for rate limiting.

    Args:
        query_per_second (float): The rate of the token bucket.
    """

    def __init__(self, rate, verbose=False):
        self._rate = rate
        self._tokens = threading.Semaphore(0)
        self.started = False
        self._request_queue = Queue()
        self.logger = get_logger()
        self.verbose = verbose

    def _add_tokens(self):
        """Add tokens to the bucket."""
        while True:
            if self._tokens._value < self._rate:
                self._tokens.release()
            sleep(1 / self._rate)

    def get_token(self):
        """Get a token from the bucket."""
        if not self.started:
            self.started = True
            threading.Thread(target=self._add_tokens, daemon=True).start()
        self._tokens.acquire()
        if self.verbose:
            cur_time = time.time()
            while not self._request_queue.empty():
                if cur_time - self._request_queue.queue[0] > 60:
                    self._request_queue.get()
                else:
                    break
            self._request_queue.put(cur_time)
            self.logger.info(f'Current RPM {self._request_queue.qsize()}.')
