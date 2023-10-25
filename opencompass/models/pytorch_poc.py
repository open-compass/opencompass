import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from transformers import AutoTokenizer

from opencompass.models.base import BaseModel
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

try:
    from lmdeploy.pytorch_poc import engine as tm
    from lmdeploy.pytorch_poc.messages import SamplingParam
except ImportError:
    from opencompass.utils import get_package_placeholder, get_placeholder
    tm = get_package_placeholder('lmdeploy')
    SamplingParam = get_placeholder('lmdeploy')

PromptType = Union[PromptList, str]


def valid_str(string, coding='utf-8'):
    """decode text according to its encoding type."""
    invalid_chars = [b'\xef\xbf\xbd']
    bstr = bytes(string, coding)
    for invalid_char in invalid_chars:
        bstr = bstr.replace(invalid_char, b'')
    ret = bstr.decode(encoding=coding, errors='ignore')
    return ret


class PytorchModel(BaseModel):
    """Model wrapper for TurboMind Python API.

    Args:
        path (str): path of the turbomind model
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
    """

    def __init__(
        self,
        path: str,
        concurrency: int = 8,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template)
        self.logger = get_logger()
        self.tokenizer = AutoTokenizer.from_pretrained(path,
                                                       trust_remote_code=True)
        tm_model = tm.Engine(path)
        self.generators = [
            tm_model.create_instance() for i in range(concurrency)
        ]
        self.generator_ids = [i + 1 for i in range(concurrency)]

    def generate(
        self,
        inputs: List[str],
        max_out_len: int = 512,
        temperature: float = 1.0,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of prompts
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic. Defaults to 1.0.

        Returns:
            List[str]: A list of generated strings.
        """
        assert isinstance(
            inputs, List), f'List(str) is expected, but got {type(inputs)}'

        # split inputs into batches
        batch_size = len(self.generators)
        batch_inputs = [
            inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)
        ]

        results = []
        for batch_input in batch_inputs:
            with ThreadPoolExecutor() as executor:
                _results = list(
                    executor.map(self._generate,
                                 self.generators[:len(batch_input)],
                                 self.generator_ids[:len(batch_input)],
                                 batch_input, [max_out_len] * len(batch_input),
                                 [temperature] * len(batch_input)))
                results += _results
        return results

    def get_token_len(self, prompt: str) -> int:
        input_ids = self.tokenizer.encode(prompt)
        return len(input_ids)

    def wait(self):
        """Wait till the next query can be sent.

        Applicable in both single-thread and multi-thread environments.
        """
        return self.token_bucket.get_token()

    def _generate(self, generator, session_id, prompt: str or PromptList,
                  max_out_len: int, temperature: float) -> str:
        """Generate results given a list of inputs.

        Args:
            prompt (str or PromptList): A string or PromptDict.
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
        assert type(
            prompt) is str, 'We only support string for TurboMind Python API'
        input_ids = self.tokenizer.encode(prompt)
        sampling_param = SamplingParam(top_k=40,
                                       top_p=0.8,
                                       temperature=temperature,
                                       repetition_penalty=1.0,
                                       ignore_eos=False,
                                       random_seed=random.getrandbits(64),
                                       stop_words=[self.eos_token_id])
        response_size = 0

        for outputs in generator.stream_infer(
                session_id=session_id,
                #   input_ids=input_ids,
                prompt_token_ids=input_ids,
                request_output_len=max_out_len,
                step=0,
                sampling_param=sampling_param):
            status, res, tokens = outputs
            response_all = self.tokenizer.decode(res)
            response_cur = response_all[response_size:]
            response_all = valid_str(response_all)
            response_size += len(response_cur)
        if hasattr(generator, 'end'):
            generator.end(session_id)
        return response_all
