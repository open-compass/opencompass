import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from queue import Queue
from typing import Dict, List, Optional, Union

import numpy as np

from opencompass.models.base import BaseModel, LMTemplateParser
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


def prepare_tensor(name, input_tensor):
    """Create grpcclient's InferInput instance according to a given tensor."""
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import np_to_triton_dtype
    t = grpcclient.InferInput(name, list(input_tensor.shape),
                              np_to_triton_dtype(input_tensor.dtype))
    t.set_data_from_numpy(input_tensor)
    return t


def stream_callback(que, result, error):
    """callback function invoked by triton client."""
    que.put((result, error))


class LmdeployTisModel(BaseModel):
    """Model wrapper for LMDeploy Python Backend Triton Inference Server gRPC
    API.

    Args:
        path (str): The name of OpenAI's model.
        tis_addr (str): The address (ip:port format) of turbomind's
            triton inference server
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
    """

    is_api: bool = True

    def __init__(self,
                 path: str,
                 tis_addr: str = '0.0.0.0:33337',
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 end_str: Optional[str] = None):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template)
        from lmdeploy.tokenizer import Tokenizer

        self.logger = get_logger()
        self.template_parser = LMTemplateParser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']
        self.tis_addr = tis_addr
        self.tokenizer = Tokenizer(path)
        self.end_str = end_str

    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len: int = 512,
        temperature: float = 1.0,
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

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs),
                             [temperature] * len(inputs),
                             [self.end_str] * len(inputs)))
        return results

    def wait(self):
        """Wait till the next query can be sent.

        Applicable in both single-thread and multi-thread environments.
        """
        return self.token_bucket.get_token()

    def get_token_len(self, prompt: str) -> int:
        input_ids = self.tokenizer.encode(prompt)
        return len(input_ids)

    def _call_triton_server(self, prompt, tis_addr, session_id,
                            request_output_len, temperature, res_que):
        import tritonclient.grpc as grpcclient

        with grpcclient.InferenceServerClient(tis_addr) as client:
            inputs = [
                prepare_tensor('prompt',
                               np.array([prompt.encode()], dtype=np.object_)),
                prepare_tensor('max_tokens',
                               np.array([request_output_len], dtype=np.int32)),
                prepare_tensor('temperature',
                               np.array([temperature], dtype=np.float_)),
                prepare_tensor('top_p', np.array([1.0], dtype=np.float_)),
                prepare_tensor('top_k', np.array([1], dtype=np.int32)),
                prepare_tensor('ignore_eos', np.array([False],
                                                      dtype=np.bool_)),
                prepare_tensor('stream', np.array([True], dtype=np.bool_)),
            ]

            # async_stream
            client.start_stream(partial(stream_callback, res_que))
            client.async_stream_infer('lmdeploy_model',
                                      inputs,
                                      sequence_id=session_id,
                                      sequence_start=True,
                                      sequence_end=True)

        res_que.put(None)
        return

    def _process_result(self, que):
        text = ''
        while True:
            res = que.get()
            if res is not None:
                result, err = res
                if err is not None:
                    print(err)
                else:
                    res = result.as_numpy('response').item().decode()
                    text += res
            else:
                return text

    def _generate(self,
                  prompt: str or PromptList,
                  max_out_len: int,
                  temperature: float,
                  end_str: Optional[str] = None) -> str:
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
            prompt
        ) is str, 'We only support string for LMDeploy Python Backend TIS API'

        res_que = Queue()

        self._call_triton_server(prompt=prompt,
                                 tis_addr=self.tis_addr,
                                 session_id=threading.currentThread().ident,
                                 request_output_len=max_out_len,
                                 temperature=temperature,
                                 res_que=res_que)
        text = self._process_result(res_que)
        response = valid_str(text)
        if end_str:
            response = response.split(end_str)[0]
        return response
