import json
import os

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.utils.prompt import PromptList

from ..base import BaseDataset


class HuProverbDataset2CQ(BaseDataset):

    @staticmethod
    def load(filepath):
        assert os.path.isfile(filepath)
        dataset = DatasetDict()
        f = open(filepath, 'r', encoding='utf-8')
        lines = f.readlines()
        out_dict_list = []
        for line in lines:
            obj = json.loads(line)
            if len(obj['context']) > 1:
                obj['context'] = '\n'.join(
                    [x.strip() for x in obj['context'] if x])
            else:
                obj['context'] = obj['context'][0]

            if obj['answer'] == 0:
                cor_ops = obj['options'][0]
                w_ops = obj['options'][1]
            else:
                cor_ops = obj['options'][1]
                w_ops = obj['options'][0]

            new_obj_1 = {
                'hu_text': obj['source_info']['proverb'],
                'context': obj['context'],
                'en_expl': obj['source_info']['en_expl'],
                'hu_expl': obj['source_info']['hu_expl'],
                'option1': cor_ops,
                'option2': w_ops,
                'out': {
                    'true_ans': '1',
                    'qid': obj['qid'],
                    'source_id': obj['source_info']['source_id'],
                    'en_expl': obj['source_info']['en_expl'],
                    'en_trans': obj['source_info']['en_trans'],
                    'hu_expl': obj['source_info']['hu_expl'],
                    'hu_text': obj['source_info']['proverb'],
                    'context': obj['context'],
                    'option1': cor_ops,
                    'option2': w_ops,
                    'correct': cor_ops,
                    'incorrect': w_ops
                }
            }

            new_obj_2 = {
                'hu_text': obj['source_info']['proverb'],
                'context': obj['context'],
                'en_expl': obj['source_info']['en_expl'],
                'hu_expl': obj['source_info']['hu_expl'],
                'option1': w_ops,
                'option2': cor_ops,
                'out': {
                    'true_ans': '2',
                    'qid': obj['qid'],
                    'source_id': obj['source_info']['source_id'],
                    'en_expl': obj['source_info']['en_expl'],
                    'en_trans': obj['source_info']['en_trans'],
                    'hu_expl': obj['source_info']['hu_expl'],
                    'hu_text': obj['source_info']['proverb'],
                    'context': obj['context'],
                    'option1': w_ops,
                    'option2': cor_ops,
                    'correct': cor_ops,
                    'incorrect': w_ops
                }
            }

            out_dict_list.append(new_obj_1)
            out_dict_list.append(new_obj_2)
        dataset = Dataset.from_list(out_dict_list)

        return dataset


class HuProverbDatasetOE(BaseDataset):

    @staticmethod
    def load(filepath):
        assert os.path.isfile(filepath)
        dataset = DatasetDict()
        f = open(filepath, 'r', encoding='utf-8')
        lines = f.readlines()
        out_dict_list = []
        for line in lines:
            obj = json.loads(line)
            if len(obj['context']) > 1:
                obj['context'] = '\n'.join(
                    [x.strip() for x in obj['context'] if x])
            else:
                obj['context'] = obj['context'][0]

            if obj['answer'] == 0:
                cor_ops = obj['options'][0]
                w_ops = obj['options'][1]
            else:
                cor_ops = obj['options'][1]
                w_ops = obj['options'][0]
            new_obj = {
                'hu_text': obj['source_info']['proverb'],
                'context': obj['context'],
                'en_expl': obj['source_info']['en_expl'],
                'hu_expl': obj['source_info']['hu_expl'],
                'out': {
                    'qid': obj['qid'],
                    'source_id': obj['source_info']['source_id'],
                    'en_expl': obj['source_info']['en_expl'],
                    'en_trans': obj['source_info']['en_trans'],
                    'hu_expl': obj['source_info']['hu_expl'],
                    'hu_text': obj['source_info']['proverb'],
                    'context': obj['context'],
                    'correct': cor_ops,
                    'incorrect': w_ops
                }
            }
            out_dict_list.append(new_obj)
        dataset = Dataset.from_list(out_dict_list)

        return dataset


class HuProverb_Evaluator_2CQ(BaseEvaluator):
    """
    ref: opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator
    """

    def score(self, predictions, references, origin_prompt) -> dict:

        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length.'}

        details = {}
        total, correct, incorrect, fail_parse = 0, 0, 0, 0
        for raw_pred, detail, ori_prompt in zip(predictions, references,
                                                origin_prompt):
            qid = detail['qid']
            option1 = detail['option1']
            option2 = detail['option2']
            true_ans = detail['true_ans']
            res_of_this_round = {
                'origin_prompt': ori_prompt,
                'is_correct': False,
                'is_incorrect': False,
                'is_fail_parse': False,
                'option1': option1,
                'option2': option2,
                'true_ans': true_ans,
                'raw_pred': raw_pred
            }
            # parse ans from raw pred
            if '1' in raw_pred and '2' not in raw_pred:
                ans = '1'
            elif '2' in raw_pred and '1' not in raw_pred:
                ans = '2'
            else:
                ans = ''
            res_of_this_round['parsed_pred'] = ans
            if ans == true_ans:
                res_of_this_round['is_correct'] = True
            elif ans == '':
                res_of_this_round['is_fail_parse'] = True
            else:
                res_of_this_round['is_incorrect'] = True

            if qid not in details:
                total += 1
                details[qid] = {
                    'detail': {
                        'hu_text': detail['hu_text'],
                        'en_trans': detail['en_trans'],
                        'en_expl': detail['en_expl'],
                        'hu_expl': detail['hu_expl'],
                        'context': detail['context'],
                        'correct': detail['correct'],
                        'incorrect': detail['incorrect']
                    },
                    'flipped_variance': [res_of_this_round],
                    'is_correct': False,
                    'is_incorrect': False,
                    'is_fail_parse': False
                }
            else:
                details[qid]['flipped_variance'].append(res_of_this_round)
                # judge the results
                if details[qid]['flipped_variance'][0][
                        'is_correct'] and details[qid]['flipped_variance'][1][
                            'is_correct']:
                    correct += 1
                    details[qid]['is_correct'] = True
                elif details[qid]['flipped_variance'][0][
                        'is_fail_parse'] or details[qid]['flipped_variance'][
                            1]['is_fail_parse']:
                    fail_parse += 1
                    details[qid]['is_fail_parse'] = True
                else:
                    incorrect += 1
                    details[qid]['is_incorrect'] = True

        assert total == correct + incorrect + fail_parse
        results = {
            'correct_ratio': correct / total * 100,
            'incorrect_ratio': incorrect / total * 100,
            'fail_parse_ratio': fail_parse / total * 100,
            'details': details
        }

        return results


class HuProverb_Evaluator_OE(BaseEvaluator):

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
        total, correct, wrong, unclear = 0, 0, 0, 0
        from opencompass.models import OpenAI
        model = OpenAI(path='gpt-4o-2024-08-06',
                       key=self.openai_key,
                       openai_proxy_url=self.openai_proxy_url,
                       max_seq_len=8192,
                       retry=2,
                       temperature=0,
                       verbose=True)
        for raw_pred, detail in zip(predictions, references):
            total += 1
            qid = detail['qid']
            details[qid] = {
                'proverb': detail['hu_text'],
                'conversation': detail['context'],
                'answer': detail['correct'],
                'raw_pred': raw_pred,
                'correctness': False,
                'ans_fail_parse': False
            }

            # ------------------------------------------- openai judge
            user_prompt = self.judge_prompt_template['en_user'].format(
                proverb=detail['hu_text'],
                conversation=detail['context'],
                answer=detail['correct'],
                raw_pred=raw_pred)
            system_prompt = self.judge_prompt_template['en_system']
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

            if 'yes' in response.lower() and 'no' not in response.lower():
                correct += 1
                details[qid]['correctness'] = True
            elif 'no' in response.lower() and 'yes' not in response.lower():
                wrong += 1
            else:
                unclear += 1
                details[qid]['ans_fail_parse'] = True

        assert total == correct + wrong + unclear
        results = {
            'correct_ratio': correct / total * 100,
            'incorrect_ratio': wrong / total * 100,
            'ans_fail_parse_ratio': unclear / total * 100,
            'details': details
        }
        return results
