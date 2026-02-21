from typing import List
from .base_api import BaseAPIModel
from openai import OpenAI
import os
import tiktoken
from opencompass.registry import MODELS
from typing import Dict, List, Optional, Union
NUM_ALLOWED_TOKENS_GPT_4 = 8192
class LLMError(Exception):
    """A custom exception used to report errors in use of Large Language Model class"""
@MODELS.register_module()
class VLLM_OPENAI(BaseAPIModel):
    is_api: bool = True
    def __init__(self,
                 path: str = "model",
                 max_seq_len: int = 2048,
                 query_per_second: int = 1,
                 retry: int = 2,
                 key: Union[str, List[str]] = 'ENV',
                 **kwargs):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         retry=retry,
                         **kwargs)
        self.model_name = path
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.retry = retry
        if isinstance(key, str):
            self.keys = [os.getenv('OPENAI_API_KEY') if key == 'ENV' else key]
        else:
            self.keys = key

    def query_vllm(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "OPENAI_API_KEY environment variable must be set when using OpenAI API."
            )
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_API_BASE'),
        )
        response = client.completions.create(  # type: ignore
            prompt=prompt,
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer: str = response.choices[0].text
        return answer

    def generate(
        self,
        inputs,
        max_out_len: int = 512,
        temperature: float = 0,
    ) -> List[str]:
        """Generate results given a list of inputs."""
        outputs = []
        for input_text in inputs:
            try:
                output = self.query_vllm(prompt=input_text, max_tokens=max_out_len, temperature=temperature)
                outputs.append(output)
            except Exception as e:
                print(f"Failed to generate output for input: {input_text} due to {e}")
        return outputs
    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string."""
        return len(self.tokenizer.encode(prompt))