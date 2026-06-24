import os
from typing import List

import anthropic

from opencompass.datasets.evalplus.provider.base import DecoderBase
# from opencompass.datasets.evalplus.provider.utility import anthropic_request


# class AnthropicDecoder(DecoderBase):
#     def __init__(self, name: str, **kwargs) -> None:
#         super().__init__(name, **kwargs)
#         self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))

#     def codegen(
#         self, prompt: str, do_sample: bool = True, num_samples: int = 200
#     ) -> List[str]:
#         if do_sample:
#             assert self.temperature > 0, "Temperature must be positive for sampling"

#         batch_size = min(self.batch_size, num_samples)
#         if not do_sample:
#             assert batch_size == 1, "Sampling only supports batch size of 1"

#         outputs = []
#         for _ in range(batch_size):
#             message = anthropic_request.make_auto_request(
#                 client=self.client,
#                 model=self.name,
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": self.instruction_prefix
#                         + f"\n```python\n{prompt.strip()}\n```\n",
#                     }
#                 ],
#                 max_tokens=self.max_new_tokens,
#                 temperature=self.temperature,
#                 stop_sequences=self.eos,
#             )
#             outputs.append(message.content[0].text)

#         return outputs

#     def is_direct_completion(self) -> bool:
#         return False
