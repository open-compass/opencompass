import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union


import requests

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel
import torch
import numpy as np

PromptType = Union[PromptList, str]


class VLLM_API(BaseAPIModel):

    def __init__(self,
                path: str,
                #  key: str,
                #  secretkey: str,
                url,
                query_per_second: int = 2,
                max_seq_len: int = 2048,
                meta_template: Optional[Dict] = None,
                retry: int = 2,
                generation_kwargs: Dict = {
                     'temperature': 0.8,
                     'prompt_logprobs': 0,
                }):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry,
                         generation_kwargs=generation_kwargs)

        self.url = url
        self.generation_kwargs = generation_kwargs

    def post_http_request(self, 
                      prompt: str,
                      api_url: str,
                      n: int = 1,
                      stream: bool = False,
                      max_out_len: int = 0):
        headers = {"User-Agent": "Test Client"}
        pload = {
            "prompt": prompt,
            "n": self.generation_kwargs['n'],
            "use_beam_search": self.generation_kwargs['use_beam_search'],
            "temperature": self.generation_kwargs['temperature'],
            "stream": stream,
            "prompt_logprobs": self.generation_kwargs['prompt_logprobs'],
            "max_tokens": self.generation_kwargs['max_out_len'],
            }
        response = requests.post(api_url, headers=headers, json=pload, 
                                 stream=True)
        return response

    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len: int = 512,
    ) -> List[str]: 

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs)))
        self.flush()
        return results

    def _generate(
        self,
        input: str or PromptList,
        max_out_len: int = 512,
    ) -> str:
        """Generate results given an input.

        Args:
            inputs (str or PromptList): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()
            try:
                raw_response = self.post_http_request(prompt=input, api_url=self.url)
                response = raw_response.json()
            except Exception as err:
                print('Request Error:{}'.format(err))
                time.sleep(3)
                continue

            self.release()

            if response is None:
                print('Connection error, reconnect.')
                # if connect error, frequent requests will casuse
                # continuous unstable network, therefore wait here
                # to slow down the request
                self.wait()
                continue
            if raw_response.status_code == 200:
                try:
                    msg = response['output']['outputs'][0]['text']
                    return msg
                except KeyError:
                    print(response)
                    self.logger.error(str(response['error_code']))
                    if response['error_code'] == 336007:
                        # exceed max length
                        return ''

                    time.sleep(1)
                    continue

            print(response)
            max_num_retries += 1

        raise RuntimeError(response['error_msg'])
   
   

    def get_ppl(self,inputs: List[str],mask_length: Optional[List[int]] = None) -> List[float]:


        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate_ppl, inputs))
        self.flush()
        results = np.array(results)
        return results


    def _generate_ppl(
        self,
        input: str or PromptList,
        # max_out_len: int = 512,
    ):

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()
            try:
                raw_response = self.post_http_request(prompt=input, api_url=self.url)
                response = raw_response.json()
            except Exception as err:
                print('Request Error:{}'.format(err))
                time.sleep(3)
                continue

            self.release()

            if response is None:
                print('Connection error, reconnect.')
                # if connect error, frequent requests will casuse
                # continuous unstable network, therefore wait here
                # to slow down the request
                self.wait()
                continue
            if raw_response.status_code == 200:
                try:
                    outputs_prob = response['output']['prompt_logprobs'][1:]
                    prompt_token_ids = response['output']['prompt_token_ids'][1:]
                    outputs_prob_list = [outputs_prob[i][str(prompt_token_ids[i])]['logprob'] for i in range(len(outputs_prob))]
                    outputs_prob_list = torch.tensor(outputs_prob_list)
                    loss = -1 * outputs_prob_list.sum(-1).cpu().detach().numpy() / len(prompt_token_ids)

                    return loss
                except KeyError:
                    print(response)
                    self.logger.error(str(response['error_code']))
                    if response['error_code'] == 336007:
                        # exceed max length
                        return ''

                    time.sleep(1)
                    continue

            print(response)
            max_num_retries += 1

        raise RuntimeError(response['error_msg'])

