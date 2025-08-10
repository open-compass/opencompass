import concurrent.futures
from typing import List

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from opencompass.models.turbomind_api import TurboMindAPIModel
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET, MODELS

from .base import BaseDataset


@LOAD_DATASET.register_module()
class OmniMathDataset(BaseDataset):

    @staticmethod
    def load():
        dataset = load_dataset('KbsdJames/Omni-MATH')['test']
        return dataset


@ICL_EVALUATORS.register_module()
class OmniMathEvaluator(BaseEvaluator):
    api_meta_template = dict(round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ])

    def __init__(self, url):
        if isinstance(url, str):
            url = [url]

        self.model = [
            MODELS.build(
                dict(
                    type=TurboMindAPIModel,
                    model_name='KbsdJames/Omni-Judge',
                    api_addr=url,
                    meta_template=self.api_meta_template,
                    temperature=0.0,
                    max_seq_len=8192,
                )) for url in url
        ]
        self.tokenizer = AutoTokenizer.from_pretrained('KbsdJames/Omni-Judge',
                                                       trust_remote_code=True)

    def batch_infer(self, models: List[TurboMindAPIModel],
                    inputs: List[str]) -> List[str]:
        batch_num = len(models)
        batch_size = (len(inputs) + batch_num - 1) // batch_num
        result_responses = []

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=batch_num) as executor:
            futures = [
                executor.submit(models[i].generate,
                                inputs[i * batch_size:(i + 1) * batch_size])
                for i in range(batch_num)
            ]
            for response in executor.map(lambda f: f.result(), futures):
                result_responses.extend(response)

        return result_responses

    def parse_response(self, response):
        response = '## Student Final Answer\n' + response.strip()

        parts = response.split('## ')
        info = {}

        for part in parts[1:]:
            lines = part.strip().split('\n')
            title = lines[0].strip()
            content = '\n'.join(lines[1:]).strip()

            if title == 'Justification':
                info[title] = content
            else:
                info[title] = lines[1].strip() if len(lines) > 1 else ''

        if info == {}:
            return False
        try:
            correctness = info['Equivalence Judgement']
            if correctness == 'TRUE':
                return True
            else:
                return False
        except Exception as e:
            print(e)
            return False

    def score(self, predictions, references, origin_prompt, test_set):
        questions = [d['problem'] for d in test_set]

        contexts = []
        for question, reference, candidate in zip(questions, references,
                                                  predictions):
            context = self.tokenizer.get_context(question, reference,
                                                 candidate)
            contexts.append(context)

        responses = self.batch_infer(self.model, contexts)
        labels = list(map(self.parse_response, responses))

        details = []
        for question, reference, candidate, response, label in zip(
                questions, references, predictions, responses, labels):
            details.append({
                'question': question,
                'reference': reference,
                'candidate': candidate,
                'response': response,
                'label': label
            })
        return {'details': details, 'accuracy': np.mean(labels) * 100}
