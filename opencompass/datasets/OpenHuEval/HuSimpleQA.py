import json
import os
import re

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.utils.prompt import PromptList

from ..base import BaseDataset


class HuSimpleQADataset(BaseDataset):

    @staticmethod
    def load(filepath):
        assert os.path.isfile(filepath)
        assert filepath.endswith('.jsonl')
        dataset = DatasetDict()
        f = open(filepath, 'r', encoding='utf-8')
        lines = f.readlines()
        objs = []
        for line in lines:
            obj = json.loads(line)
            objs.append(obj)
        out_dict_list = []
        for obj in objs:
            question = obj['question']
            hu_specific_dim = obj['hu_specific_dim']
            tmp = obj
            new_obj = dict(question=question,
                           hu_specific_dim=hu_specific_dim,
                           reference=tmp)
            out_dict_list.append(new_obj)
        dataset = Dataset.from_list(out_dict_list)
        return dataset


class HuSimpleQAEvaluator(BaseEvaluator):

    def __init__(self,
                 judge_prompt_template,
                 openai_key='ENV',
                 openai_proxy_url='ENV',
                 **kwargs):
        super().__init__(**kwargs)
        self.judge_prompt_template = judge_prompt_template
        self.openai_key = openai_key
        self.openai_proxy_url = openai_proxy_url

    def score(self, predictions, references, origin_prompt) -> dict:
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length.'}

        details = {}
        total, correct, wrong, not_attempted, failed_to_parse = 0, 0, 0, 0, 0
        from opencompass.models import OpenAI
        model = OpenAI(path='gpt-4o-2024-08-06',
                       key=self.openai_key,
                       openai_proxy_url=self.openai_proxy_url,
                       max_seq_len=8192,
                       retry=2,
                       temperature=0,
                       verbose=True)

        confidence_scores = []
        for raw_pred, detail in zip(predictions, references):
            total += 1
            qid = detail['qid']
            details[qid] = {
                'question': detail['question'],
                'answer': detail['answer'],
                'raw_pred': raw_pred,
                'correctness': False,
                'failed_to_parse': False
            }
            # parse raw_pred
            try:
                raw_pred = re.sub(r'^```json\n|\n```$', '', raw_pred)
                raw_pred_json = json.loads(raw_pred)
                confidence_score = raw_pred_json.get('confidence_score', None)
            except json.JSONDecodeError:
                confidence_score = None
            details[qid]['confidence_score'] = confidence_score

            # ------------------------ involve openai gpt4o as judge
            user_prompt = self.judge_prompt_template['user_prompt'].format(
                question=detail['question'],
                answer=detail['answer'],
                pred_answer=raw_pred)
            system_prompt = self.judge_prompt_template['system_prompt']
            details[qid]['judge_user_prompt'] = user_prompt

            messages = PromptList([{
                'role': 'SYSTEM',
                'prompt': system_prompt,
            }, {
                'role': 'HUMAN',
                'prompt': user_prompt,
            }])
            response = model._generate(input=messages,
                                       max_out_len=8192,
                                       temperature=0.1)
            details[qid]['judge_resp'] = response
            try:
                response = re.sub(r'^```json\n|\n```$', '', response)
                evaluation_result = json.loads(response)
                evaluation = evaluation_result.get('evaluation', '').lower()

                details[qid]['correctness'] = (evaluation == 'correct')
                details[qid]['failed_to_parse'] = False

                if evaluation == 'correct':
                    correct += 1
                elif evaluation == 'incorrect':
                    wrong += 1
                elif evaluation == 'not_attempted':
                    not_attempted += 1
                else:
                    failed_to_parse += 1

            except json.JSONDecodeError:
                details[qid]['failed_to_parse'] = True
                failed_to_parse += 1

            confidence_scores.append(
                (confidence_score, details[qid]['correctness']))

        accuracy = correct / total if total > 0 else 0

        results = {
            'accuracy': accuracy,
            'total': total,
            'correct': correct,
            'wrong': wrong,
            'not_attempted': not_attempted,
            'failed_to_parse': failed_to_parse,
            'details': details,
            'confidence_scores': confidence_scores
        }
        return results
