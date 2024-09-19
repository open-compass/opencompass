import json
import time
from logging import getLogger

import requests
from openai import OpenAI

from .xfinder_utils import PROMPT_TEMPLATE

Instruction = """I will provide you with a question, output sentences along with an answer range. The output sentences are the response of the question provided. The answer range could either describe the type of answer expected or list all possible valid answers. Using the information provided, you must accurately and precisely determine and extract the intended key answer from the output sentences. Please don't have your subjective thoughts about the question.
First, you need to determine whether the content of the output sentences is relevant to the given question. If the entire output sentences are unrelated to the question (meaning the output sentences are not addressing the question), then output [No valid answer].
Otherwise, ignore the parts of the output sentences that have no relevance to the question and then extract the key answer that matches the answer range.
Below are some special cases you need to be aware of:
    (1) If the output sentences present multiple different answers, carefully determine if the later provided answer is a correction or modification of a previous one. If so, extract this corrected or modified answer as the final response. Conversely, if the output sentences fluctuate between multiple answers without a clear final answer, you should output [No valid answer].
    (2) If the answer range is a list and the key answer in the output sentences is not explicitly listed among the candidate options in the answer range, also output [No valid answer].

""" # noqa


class Extractor:

    def __init__(
        self,
        model_name,
        model_path=None,
        url=None,
        temperature=0,
        max_tokens=3000,
        api_key='EMPTY',
        SYSTEM='You are a help assistant tasked with extracting the precise key answer from given output sentences. You must only provide the extracted key answer without including any additional text.'  # noqa
    ):
        self.model_name = model_name
        self.PROMPT_TEMPLATE = PROMPT_TEMPLATE[model_name]
        self.SYSTEM = SYSTEM
        self.model_path = model_path
        self.url = url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mode = 'API' if self.url is not None else 'Local'
        self.logger = getLogger(__name__)

        if self.mode == 'Local':
            from vllm import LLM, SamplingParams
            self.sampling_params = SamplingParams(temperature=self.temperature,
                                                  max_tokens=self.max_tokens,
                                                  stop=[
                                                      '<|endoftext|>',
                                                      '<|im_end|>', '<eoa>',
                                                      '<||>', '<end_of_turn>',
                                                      '<|eot_id|>'
                                                  ])
            self.llm = LLM(model=self.model_path, gpu_memory_utilization=0.5)

    @staticmethod
    def prepare_input(item):
        user_input = Instruction + \
            "Question: \"\"\"" + item['question'] + "\"\"\"\n\n" + \
            "Output sentences: \"\"\"" + item['llm_output'] + "\"\"\"\n\n" + \
            'Answer range: ' + item['standard_answer_range'] + '\n\n' + \
            'Key extracted answer: '

        return user_input

    def gen_output(self, query):
        if self.mode == 'API':
            # return self.send_request(query)
            return self.openai_infer(query)
        else:
            return self.offline_infer(query)

    def send_request(self, query: str) -> str:
        """Send a request to the model's API and return the response.

        Args:
            query (str): The input query.

        Returns:
            str: The extracted answer (xFinder's output).
        """
        prompt = self.PROMPT_TEMPLATE.format(system=self.SYSTEM, input=query)
        payload = json.dumps({
            'prompt':
            prompt,
            'temperature':
            self.temperature,
            'max_tokens':
            self.max_tokens,
            'stop': [
                '<|endoftext|>', '<|im_end|>', '<eoa>', '<||>',
                '<end_of_turn>', '<|eot_id|>'
            ],
        })
        headers = {'Content-Type': 'application/json'}
        res = requests.request('POST', self.url, headers=headers, data=payload)
        res = res.json()['text'][0]
        res = res.replace(prompt, '')
        # res = requests.post(self.url, json=payload)
        # res = res.json()['text']
        res = res.strip()
        return res

    def openai_infer(self, query: str, retry=9) -> str:
        """Perform inference on the OpenAI model.

        Args:
            query (str): The input query.

        Returns:
            str: The extracted answer (xFinder's output).
        """
        if isinstance(self.url, list):
            # Randomly api for better load balancing
            import random
            self.url = random.choice(self.url)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.url,
        )
        self.retry = retry

        t = time.time()
        retry = self.retry
        response = ''
        while retry > 0:
            try:
                chat_response = self.client.chat.completions.create(
                    model=self.client.models.list().data[0].id
                    if self.model_name == '' else self.model_name,
                    messages=[
                        {
                            'role': 'system',
                            'content': self.SYSTEM
                        },
                        {
                            'role': 'user',
                            'content': query
                        },
                    ],
                    stop=[
                        '<|endoftext|>', '<|im_end|>', '<eoa>', '<||>',
                        '<end_of_turn>', '<|eot_id|>'
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                js_response = json.loads(chat_response.model_dump_json())
                response = js_response['choices'][0]['message']['content']
                break
            except Exception as e:
                self.logger.info(f'Error: {e}')
                self.logger.info(f'{self.url} is down. Retrying...')
                self.logger.info(f'Time elapsed: {time.time() - t} seconds')
                time.sleep(6)
                retry -= 1
        if retry == 0:
            response = 'Error: Failed to get response.'
            self.logger.info(f'{response} after {self.retry} tries.')
            raise ValueError('The api is down')
        return response.strip()

    def offline_infer(self, query: str) -> str:
        """Perform inference on the local xFinder model.

        Args:
            query (str): The input query.

        Returns:
            str: The extracted answer (xFinder's output).
        """
        prompt = self.PROMPT_TEMPLATE.format(system=self.SYSTEM, input=query)
        res = self.llm.generate(prompt, self.sampling_params)
        res = res[0]
        res = res.outputs[0].text.strip()
        return res
