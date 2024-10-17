import requests
from typing import List
from .base_api import BaseAPIModel
import fireworks.client
import os
import tiktoken
from opencompass.registry import MODELS
from typing import Dict, List, Optional, Union
NUM_ALLOWED_TOKENS_GPT_4 = 8192
# Load environment variables
fireworks.client.api_key = os.getenv('FIREWORKS_API_KEY')
class LLMError(Exception):
    """A custom exception used to report errors in use of Large Language Model class"""
@MODELS.register_module()
class Fireworks(BaseAPIModel):
    is_api: bool = True
    def __init__(self,
                 path: str = "accounts/fireworks/models/mistral-7b",
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
    def query_fireworks(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.01,
    ) -> str:
        if max_tokens == 0:
            max_tokens = NUM_ALLOWED_TOKENS_GPT_4  # Adjust based on your model's capabilities
        completion = fireworks.client.Completion.create(
            model=self.model_name,
            prompt=prompt,
            n=1,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if len(completion.choices) != 1:
            raise LLMError("Unexpected number of choices returned by Fireworks.")
        return completion.choices[0].text
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
                output = self.query_fireworks(prompt=input_text, max_tokens=max_out_len, temperature=temperature)
                outputs.append(output)
            except Exception as e:
                print(f"Failed to generate output for input: {input_text} due to {e}")
        return outputs
    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string."""
        return len(self.tokenizer.encode(prompt))