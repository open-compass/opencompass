import os.path as osp
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from opencompass.models.base import BaseModel
from opencompass.models.base_api import TokenBucket
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


def valid_str(string, coding='utf-8'):
    """decode text according to its encoding type."""
    invalid_chars = [b'\xef\xbf\xbd']
    bstr = bytes(string, coding)
    for invalid_char in invalid_chars:
        bstr = bstr.replace(invalid_char, b'')
    ret = bstr.decode(encoding=coding, errors='ignore')
    return ret


class TurboMindModel(BaseModel):
    """Model wrapper for TurboMind API.

    Args:
        path (str): The name of OpenAI's model.
        model_path (str): folder of the turbomind model's path
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
    """

    is_api: bool = True

    def __init__(
        self,
        path: str,
        model_path: str,
        max_seq_len: int = 2048,
        query_per_second: int = 1,
        retry: int = 2,
        meta_template: Optional[Dict] = None,
    ):

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template)
        self.logger = get_logger()

        from lmdeploy import turbomind as tm
        from lmdeploy.model import MODELS as LMMODELS
        from lmdeploy.turbomind.tokenizer import Tokenizer as LMTokenizer

        self.retry = retry

        tokenizer_model_path = osp.join(model_path, 'triton_models',
                                        'tokenizer')
        self.tokenizer = LMTokenizer(tokenizer_model_path)
        tm_model = tm.TurboMind(model_path, eos_id=self.tokenizer.eos_token_id)
        self.model_name = tm_model.model_name
        self.model = LMMODELS.get(self.model_name)()
        self.generator = tm_model.create_instance()
        self.token_bucket = TokenBucket(query_per_second)

    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len: int = 512,
        temperature: float = 0.0,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str or PromptList]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic. Defaults to 0.7.

        Returns:
            List[str]: A list of generated strings.
        """
        prompts = inputs
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, prompts,
                             [max_out_len] * len(inputs),
                             [temperature] * len(inputs)))
        return results

    def wait(self):
        """Wait till the next query can be sent.

        Applicable in both single-thread and multi-thread environments.
        """
        return self.token_bucket.get_token()

    def _generate(self, input: str or PromptList, max_out_len: int,
                  temperature: float) -> str:
        """Generate results given a list of inputs.

        Args:
            inputs (str or PromptList): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))

        assert type(
            input
        ) is str, 'We only support string for TurboMind Python API now'

        intput_token_ids = self.tokenizer.encode(input)

        for _ in range(self.retry):
            self.wait()
            session_id = random.randint(1, 100000)
            nth_round = 0
            for outputs in self.generator.stream_infer(
                    session_id=session_id,
                    input_ids=[intput_token_ids],
                    stream_output=False,
                    request_output_len=max_out_len,
                    sequence_start=(nth_round == 0),
                    sequence_end=False,
                    step=0,
                    stop=False,
                    top_k=40,
                    top_p=0.8,
                    temperature=temperature,
                    repetition_penalty=1.0,
                    ignore_eos=False,
                    random_seed=random.getrandbits(64)
                    if nth_round == 0 else None):
                pass

        output_token_ids, _ = outputs[0]
        # decode output_token_ids
        response = self.tokenizer.decode(output_token_ids)
        response = valid_str(response)

        return response
