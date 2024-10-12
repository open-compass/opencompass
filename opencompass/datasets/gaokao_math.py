import concurrent.futures
import json
import re

from datasets import Dataset

from opencompass.models import OpenAISDK
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET, MODELS

from .base import BaseDataset

# from opencompass.utils import get_data_path


EVAL_PROMPT = """
请你作为一个数学高考阅卷专家，判断下面的答案是否与标准答案一致，即考生是否回答正确。下面是一些评判标准：
1. 有些答案可能包含多项内容，可能有单选题，多选题，填空题等，只要答案与标准答案一致即可, 对于多选题和多个空的填空题，需要考生对应的选项或空都回答正确才算正确。
2. 有些答案可能通过不同的方式表达，比如有些答案可能是一个数学表达式，有些答案可能是一个文字描述，只要表达的意思一致即可。且有些公式通过不同的方式表达，但等价，也是正确的。
3. 你不需要重新计算问题答案，因为标准答案已经给出，只需要根据问题形式来判断考生的答案是否与标准答案一致，是否正确即可。

请你根据上述标准，判断下面的答案是否与标准答案一致，如果一致，请在最后输出\\boxed{{yes}}, 否则输出\\boxed{{no}}, 如果难以判断，请输出\\boxed{{no}}.
原问题：{question}
标准答案：{gold_answer}
考生答案：{answer}

分析：
""" # noqa E501


def extract_boxed_answer(text):
    match = re.findall(r'\\boxed{(.+?)}', text)
    if match:
        return match[-1]
    return None


@LOAD_DATASET.register_module()
class GaoKaoMATHDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        # path = get_data_path(path, local_mode=True)
        data = json.load(open(path))
        for i in range(len(data)):
            data[i]['extract_answer'] = str(data[i]['extract_answer'])
        dataset = Dataset.from_list(data)
        return dataset


api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])


@ICL_EVALUATORS.register_module()
class GaoKaoMATHEvaluator(BaseEvaluator):

    def __init__(self, model_name, url, **kwargs):
        if isinstance(url, str):
            url = [url]

        self.model = [
            MODELS.build(
                dict(
                    type=OpenAISDK,
                    path=model_name,
                    openai_api_base=url,
                    key='EMPTY',
                    query_per_second=1,
                    meta_template=api_meta_template,
                    temperature=kwargs.get('temperature', 0.01),
                    max_seq_len=kwargs.get('max_tokens', 8192),
                )) for url in url
        ]

    def batch_response(self, inputs):
        batch_num = len(self.model)
        batch_size = (len(inputs) + batch_num - 1) // batch_num
        result_responses = []

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=batch_num) as executor:
            futures = [
                executor.submit(self.model[i].generate,
                                inputs[i * batch_size:(i + 1) * batch_size])
                for i in range(batch_num)
            ]
            for response in executor.map(lambda f: f.result(), futures):
                result_responses.extend(response)

        return result_responses

    def score(self, predictions, references, origin_prompt):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        questions = [item[0]['prompt'] for item in origin_prompt]
        count = 0
        correct = 0
        details = []
        results = []
        inputs = []
        for pred, ref, ques in zip(predictions, references, questions):
            inputs.append(
                EVAL_PROMPT.format(answer=pred, gold_answer=ref,
                                   question=ques))

        result_responses = self.batch_response(inputs)
        results = [
            extract_boxed_answer(result) == 'yes'
            for result in result_responses
        ]
        for pred, ref, result, result_response in zip(predictions, references,
                                                      results,
                                                      result_responses):
            detail = {
                'pred': pred,
                'answer': ref,
                'correct': False,
                'eval_model_response': result_response
            }
            count += 1
            if result:
                correct += 1
                detail['correct'] = True
            details.append(detail)

        detailed_result = {
            'accuracy': 100 * correct / count,
            'details': details
        }

        return detailed_result


if __name__ == '__main__':
    evaluator = GaoKaoMATHEvaluator('http://0.0.0.0:23333/v1',
                                    temperature=0.01,
                                    max_tokens=2048,
                                    procs=8)
    predictions = ['1', '2', '3']
    references = ['1', '2', '3']
    evaluator.score(predictions, references)
