import json
import re
import time
from logging import getLogger

from datasets import Dataset
from openai import OpenAI

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

EVAL_PROMPT = """
请你作为一个数学高考阅卷专家，判断下面的答案是否与标准答案一致，即考生是否回答正确。下面是一些评判标准：
1. 有些答案可能包含多项内容，可能有单选题，多选题，填空题等，只要答案与标准答案一致即可, 对于多选题和多个空的填空题，需要考生对应的选项或空都回答正确才算正确。
2. 有些答案可能通过不同的方式表达，比如有些答案可能是一个数学表达式，有些答案可能是一个文字描述，只要表达的意思一致即可。且有些公式通过不同的方式表达，但等价，也是正确的。

请你根据上述标准，判断下面的答案是否与标准答案一致，如果一致，请在最后输出\\boxed{yes}, 否则输出\\boxed{no}, 如果难以判断，请输出\\boxed{no}.
考生答案：{answer}
标准答案：{gold_answer}

分析：
""" # noqa E501


def extract_boxed_answer(text):
    match = re.search(r'\\boxed{(.+)}', text)
    if match:
        return match.group(1)
    return None


@LOAD_DATASET.register_module()
class GaoKaoMATHDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        data = json.load(open(path))
        dataset = Dataset.from_list(data)
        return dataset


class API_Infer:

    def __init__(self, api_key, url, model_name, temperature, max_tokens):
        self.api_key = api_key
        self.url = url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.SYSTEM = 'You are a helpful assistant.'
        self.logger = getLogger(__name__)

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


@ICL_EVALUATORS.register_module()
class GaoKaoMATHEvaluator(BaseEvaluator):

    def __init__(self,
                 url,
                 temperature=1e-6,
                 max_tokens=2048,
                 procs=8,
                 **kwargs):
        self.model = API_Infer('', url, '', temperature, max_tokens)
        self.procs = procs

    def is_equiv(self, i, j):
        judges = []
        for pred, ref in zip(i, j):
            pred = self.model.openai_infer(
                EVAL_PROMPT.replace('{answer}',
                                    pred).replace('{gold_answer}', ref))
            if extract_boxed_answer(pred) == 'yes':
                judges.append(1)
            else:
                judges.append(0)
        return judges

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        details = []
        correct = 0
        count = 0
        results = []
        for pred, ref in zip(predictions, references):
            result = self.is_equiv(pred, ref)
            results.append(result)

        for pred, ref, result in zip(predictions, references, results):
            detail = {'pred': pred, 'answer': ref, 'correct': False}
            count += 1
            if result:
                correct += 1
                detail['correct'] = True
            details.append(detail)

        detailed_result = {
            'accuracy': 100 * correct / count,
            'details': details
        }
        self.logger.info(json.dumps(detailed_result, indent=4))
        return detailed_result


if __name__ == '__main__':
    evaluator = GaoKaoMATHEvaluator('http://22.8.75.210:23333/v1',
                                    temperature=0.01,
                                    max_tokens=2048,
                                    procs=8)
    predictions = ['1', '2', '3']
    references = ['1', '2', '3']
    evaluator.score(predictions, references)
